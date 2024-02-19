//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::{collections::VecDeque, hash::Hash};

use bimap::BiMap;
use portgraph::{LinkView, NodeIndex, PortOffset};

use crate::{
    patterns::{Edge, UnweightedEdge},
    utils::{always_true, validate_unweighted_edge},
    EdgeProperty, NodeProperty, Pattern, PatternID, Universe,
};

use super::{Match, PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<U: Universe, PNode, PEdge: Eq + Hash> {
    pattern: Pattern<U, PNode, PEdge>,
    edges: Vec<Edge<U, PNode, PEdge>>,
    root: U,
}

impl<U, G> PortMatcher<G, NodeIndex, U> for SinglePatternMatcher<U, (), UnweightedEdge>
where
    G: LinkView + Copy,
    U: Universe,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: G, root: NodeIndex) -> Vec<Match> {
        self.find_rooted_match(root, always_true, validate_unweighted_edge(graph))
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

impl<U, PNode, PEdge> SinglePatternMatcher<U, PNode, PEdge>
where
    U: Universe,
    PNode: NodeProperty,
    PEdge: EdgeProperty,
{
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    fn match_exists<N: Universe>(
        &self,
        host_root: N,
        validate_node: impl for<'a> Fn(N, &PNode) -> bool,
        validate_edge: impl for<'a> Fn(N, &'a PEdge) -> Vec<Option<N>>,
    ) -> bool {
        !self
            .get_match_map(host_root, validate_node, validate_edge)
            .is_empty()
    }

    /// Match the pattern and return a map from pattern nodes to host nodes
    ///
    /// Returns `None` if the pattern does not match.
    pub fn get_match_map<N: Universe>(
        &self,
        host_root: N,
        validate_node: impl for<'a> Fn(N, &PNode) -> bool,
        validate_edge: impl for<'a> Fn(N, &'a PEdge) -> Vec<Option<N>>,
    ) -> Vec<BiMap<U, N>> {
        let mut candidates = VecDeque::new();
        candidates.push_back((
            self.edges.as_slice(),
            BiMap::from_iter([(self.root, host_root)]),
        ));
        let mut final_match_maps = Vec::new();
        while let Some((edges, match_map)) = candidates.pop_front() {
            let Some(e) = edges.first() else {
                final_match_maps.push(match_map);
                continue;
            };
            let edges = &edges[1..];

            let src = e.source.expect("Only connected edges allowed in pattern");
            let tgt = e.target.expect("Only connected edges allowed in pattern");
            let Some(&new_src) = match_map.get_by_left(&src) else {
                continue;
            };
            let new_tgts = validate_edge(new_src, &e.edge_prop).into_iter().flatten();
            for new_tgt in new_tgts {
                let mut new_match_map = match_map.clone();
                if let Some(target_prop) = e.target_prop.as_ref() {
                    if !validate_node(new_tgt, target_prop) {
                        continue;
                    }
                }
                if match_map.get_by_left(&tgt) != Some(&new_tgt) {
                    let Ok(_) = new_match_map.insert_no_overwrite(tgt, new_tgt) else {
                        continue;
                    };
                }
                candidates.push_back((edges, new_match_map));
            }
        }
        final_match_maps
    }

    /// The matches in `host` starting at `host_root`
    ///
    /// For single pattern matchers there is always at most one match
    fn find_rooted_match<N: Universe>(
        &self,
        host_root: N,
        validate_node: impl for<'a> Fn(N, &PNode) -> bool,
        validate_edge: impl for<'a> Fn(N, &'a PEdge) -> Vec<Option<N>>,
    ) -> Vec<PatternMatch<PatternID, N>> {
        if self.match_exists(host_root, validate_node, validate_edge) {
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
    pub fn into_single_pattern_matcher(
        self,
    ) -> SinglePatternMatcher<U, (), (PortOffset, PortOffset)> {
        SinglePatternMatcher::new(self)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::{LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView};

    use crate::{matcher::PortMatcher, utils::test::graph, Pattern};

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
                .flat_map(|m| m.to_match_map(&g, &matcher))
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
            .flat_map(|m| m.to_match_map(&g, &matcher))
            .map(|m| m.values().sorted().copied().collect_vec())
            .collect_vec();
        matches.sort_unstable_by_key(|v| *v.first().unwrap());
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], vs1.to_vec());
        assert_eq!(matches[1], vs2.to_vec());
        assert_eq!(matches[2], vs3.to_vec());
    }
}
