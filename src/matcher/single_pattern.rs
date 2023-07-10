//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::{borrow::Borrow, hash::Hash};

use bimap::{BiMap};
use portgraph::{LinkView, NodeIndex, PortGraph, PortOffset, PortView};

use crate::{
    patterns::{Edge, UnweightedEdge},
    EdgeProperty, NodeProperty, Pattern, PatternID, Universe,
};

use super::{Match, PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<U: Universe, PNode, PEdge: Eq + Hash> {
    pattern: Pattern<U, PNode, PEdge>,
    edges: Vec<Edge<U, PNode, PEdge>>,
    root: U,
}

impl<U> PortMatcher<PortGraph, U> for SinglePatternMatcher<U, (), UnweightedEdge>
where
    U: Universe,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Match<PortGraph>> {
        self.find_rooted_match(graph, root, validate_unweighted_edge)
    }

    fn get_pattern(&self, _id: crate::PatternID) -> Option<&Pattern<U, Self::PNode, Self::PEdge>> {
        Some(&self.pattern)
    }
}

// TODO: add impls of PortMatcher for weighted graphs etc

impl<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty> SinglePatternMatcher<U, PNode, PEdge> {
    /// Create a new matcher for a single pattern.
    pub fn new(pattern: Pattern<U, PNode, PEdge>) -> Self {
        // This is our "matching recipe" -- we precompute it once and store it
        let edges = pattern.edges().expect("Cannot match disconnected pattern");
        let root = pattern.root().expect("Cannot match unrooted pattern");
        Self {
            pattern,
            edges,
            root,
        }
    }

    pub fn from_pattern(pattern: Pattern<U, PNode, PEdge>) -> Self {
        Self::new(pattern)
    }
}

impl<U, PNode, PEdge: Eq + Hash> SinglePatternMatcher<U, PNode, PEdge>
where
    U: Universe,
    PNode: Copy,
    PEdge: Copy,
{
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    fn match_exists<G, V, F>(&self, host: G, host_root: V, validate_edge: F) -> bool
    where
        F: Fn(Edge<V, PNode, PEdge>, G) -> Option<(V, V)>,
        V: Universe,
        G: Copy,
    {
        self.get_match_map(host, host_root, validate_edge).is_some()
    }

    pub(crate) fn get_match_map<G, V, F>(
        &self,
        host: G,
        host_root: V,
        validate_edge: F,
    ) -> Option<BiMap<U, V>>
    where
        F: Fn(Edge<V, PNode, PEdge>, G) -> Option<(V, V)>,
        V: Universe,
        G: Copy,
    {
        let mut match_map = BiMap::from_iter([(self.root, host_root)]);
        for &e in self.edges.iter() {
            let src = e.source.expect("Only connected edges allowed in pattern");
            let tgt = e.target.expect("Only connected edges allowed in pattern");
            let e_in_v = Edge {
                source: match_map.get_by_left(&src).copied(),
                target: match_map.get_by_left(&tgt).copied(),
                edge_prop: e.edge_prop,
                source_prop: e.source_prop,
                target_prop: e.target_prop,
            };
            let (new_src, new_tgt) = validate_edge(e_in_v, host)?;
            if !match_map.contains_left(&src) {
                match_map.insert_no_overwrite(src, new_src).ok()?;
            }
            if !match_map.contains_left(&tgt) {
                match_map.insert_no_overwrite(tgt, new_tgt).ok()?;
            }
        }
        Some(match_map)
    }

    /// The matches in `host` starting at `host_root`
    ///
    /// For single pattern matchers there is always at most one match
    fn find_rooted_match<G, V, F>(
        &self,
        host: G,
        host_root: V,
        validate_edge: F,
    ) -> Vec<PatternMatch<PatternID, V>>
    where
        F: Fn(Edge<V, PNode, PEdge>, G) -> Option<(V, V)>,
        V: Universe,
        G: Copy,
    {
        if self.match_exists(host, host_root, validate_edge) {
            vec![PatternMatch {
                pattern: 0.into(),
                root: host_root,
            }]
        } else {
            Vec::new()
        }
    }
}

impl<U: Universe> Pattern<U, (), (PortOffset, PortOffset)> {
    pub fn into_single_pattern_matcher(self) -> impl PortMatcher<PortGraph, U> {
        SinglePatternMatcher::new(self)
    }
}

/// Check if an edge `e` is valid in a portgraph `g` without weights.
pub(crate) fn validate_unweighted_edge<G>(
    e: Edge<NodeIndex, (), UnweightedEdge>,
    g: G,
) -> Option<(NodeIndex, NodeIndex)>
where
    G: Borrow<PortGraph>,
{
    let g = g.borrow();
    let mut flipped = false;
    let src = e.source;
    let tgt = e.target;
    let (src, tgt) = if src.is_none() {
        flipped = true;
        (tgt.expect("both source and target are none"), src)
    } else {
        (src.unwrap(), tgt)
    };
    let (src_port, should_tgt_port) = if flipped {
        (e.edge_prop.1, e.edge_prop.0)
    } else {
        e.edge_prop
    };
    let src_port = g.port_index(src, src_port)?;
    let tgt_port = g.port_link(src_port)?;
    if let Some(tgt) = tgt {
        if tgt != g.port_node(tgt_port).unwrap() {
            return None;
        }
    }
    let tgt = g.port_node(tgt_port).unwrap();
    if g.port_offset(tgt_port).unwrap() != should_tgt_port {
        return None;
    }
    if flipped {
        Some((tgt, src))
    } else {
        Some((src, tgt))
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::{
        LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView,
    };
    

    use crate::{
        matcher::PortMatcher,
        utils::test::{graph},
        Pattern,
    };

    use super::SinglePatternMatcher;

    #[test]
    fn single_pattern_match_simple() {
        let g = graph();
        let p = Pattern::from_portgraph(&g);
        let matcher = SinglePatternMatcher::from_pattern(p);

        let (n0, n1, n3, n4) = (
            NodeIndex::new(0),
            NodeIndex::new(1),
            NodeIndex::new(3),
            NodeIndex::new(4),
        );
        assert_eq!(
            matcher
                .find_matches(&g)
                .into_iter()
                .map(|m| m.to_match_map(&g, &matcher).unwrap())
                .collect_vec(),
            vec![[(n0, n0), (n1, n1), (n3, n3), (n4, n4)]
                .into_iter()
                .collect()]
        );
    }

    #[test]
    fn single_pattern_single_node() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_portgraph(&g);
        let matcher = SinglePatternMatcher::from_pattern(p);
        let mut g = PortGraph::new();
        g.add_node(1, 0);

        assert_eq!(matcher.find_matches(&g).len(), 1);
    }

    #[test]
    fn single_node_loop() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_portgraph(&g);
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        let n = g.add_node(2, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g).len(), 1);
    }

    #[test]
    fn single_node_loop_no_match() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_portgraph(&g);
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n1, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    fn add_pattern(graph: &mut PortGraph, vertices: &[NodeIndex; 4]) {
        let [_, _, v2_in, v3_in] = vertices.map(|n| graph.inputs(n).collect_vec());
        let [v0_out, v1_out, v2_out, _] = vertices.map(|n| graph.outputs(n).collect_vec());

        graph.link_ports(v0_out[0], v2_in[1]).unwrap();
        graph.link_ports(v1_out[1], v2_in[0]).unwrap();
        graph.link_ports(v2_out[0], v3_in[1]).unwrap();
        graph.link_ports(v1_out[2], v3_in[2]).unwrap();
    }

    #[test]
    fn single_pattern_match_complex() {
        let mut pattern = PortGraph::new();
        for _ in 0..4 {
            pattern.add_node(3, 3);
        }
        let pi = |i| pattern.nodes_iter().nth(i).unwrap();
        let ps = [pi(0), pi(1), pi(2), pi(3)];
        add_pattern(&mut pattern, &ps);
        let p = Pattern::from_portgraph(&pattern);
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        for _ in 0..100 {
            g.add_node(3, 3);
        }
        let vi = |i| g.nodes_iter().nth(i).unwrap();
        let vs1 = [vi(0), vi(10), vi(30), vi(55)];
        let vs2 = [vi(3), vi(12), vi(23), vi(44)];
        let vs3 = [vi(12), vi(55), vi(98), vi(99)];
        add_pattern(&mut g, &vs1);
        add_pattern(&mut g, &vs2);
        add_pattern(&mut g, &vs3);

        let mut matches = matcher
            .find_matches(&g)
            .into_iter()
            .map(|m| {
                m.to_match_map(&g, &matcher)
                    .unwrap()
                    .values()
                    .sorted()
                    .copied()
                    .collect_vec()
            })
            .collect_vec();
        matches.sort_unstable_by_key(|v| *v.first().unwrap());
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], vs1.to_vec());
        assert_eq!(matches[1], vs2.to_vec());
        assert_eq!(matches[2], vs3.to_vec());
    }
}
