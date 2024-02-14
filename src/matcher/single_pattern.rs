//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.

use std::hash::Hash;

use crate::{
    constraint::{ScopeConstraint, ScopeMap},
    pattern,
    portgraph::RootedPortMatcher,
    HashMap, Pattern, PatternID, Symbol,
};

use super::PatternMatch;

use delegate::delegate;

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<P: Pattern>(P);

impl<P: Pattern> SinglePatternMatcher<P>
where
    P::Universe: Eq + Hash,
{
    delegate! {
        to self.0 {
            pub fn constraints(&self) -> impl Iterator<Item = P::Constraint> + '_;
            pub fn id(&self) -> PatternID;
        }
    }

    /// Create a new matcher for a single pattern.
    pub fn new(pattern: P) -> Self {
        Self(pattern)
    }

    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    fn match_exists(&self, host_root: pattern::Value<P>, graph: pattern::DataRef<P>) -> bool {
        self.get_match_map(host_root, graph).is_some()
    }

    /// Match the pattern and return a map from pattern nodes to host nodes
    ///
    /// Returns `None` if the pattern does not match.
    pub fn get_match_map(
        &self,
        host_root: pattern::Value<P>,
        graph: pattern::DataRef<P>,
    ) -> Option<HashMap<P::Universe, pattern::Value<P>>> {
        let mut scope = pattern::ScopeMap::<P>::from_iter([(
            <pattern::Symbol<P> as Symbol>::root(),
            host_root,
        )]);
        for c in self.constraints() {
            let new_symbols = c.is_satisfied(graph, &scope)?;
            for (symb, val) in new_symbols {
                if scope.get_by_left(&symb) == Some(&val) {
                    continue;
                }
                scope.insert_no_overwrite(symb, val).ok()?;
            }
        }

        let mut match_map = HashMap::default();
        for (symb, val) in scope {
            match_map.insert(self.0.get_id(symb).expect("symbol not found in scope"), val);
        }
        Some(match_map)
    }

    /// The matches in `host` starting at `host_root`
    ///
    /// For single pattern matchers there is always at most one match
    pub(crate) fn find_rooted_match(
        &self,
        host_root: pattern::Value<P>,
        graph: pattern::DataRef<P>,
    ) -> Vec<PatternMatch<PatternID, pattern::Value<P>>> {
        if self.match_exists(host_root, graph) {
            vec![PatternMatch {
                pattern: 0.into(),
                root: host_root,
            }]
        } else {
            Vec::new()
        }
    }

    pub fn pattern(&self) -> &P {
        &self.0
    }
}

impl<P: Pattern> From<P> for SinglePatternMatcher<P>
where
    P::Universe: Eq + Hash,
{
    fn from(value: P) -> Self {
        SinglePatternMatcher::new(value)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::{LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView};

    use crate::{utils::test::graph, Pattern};

    use crate::portgraph::{PortMatcher, PortgraphPattern, PortgraphPatternBuilder};

    use super::SinglePatternMatcher;

    use petgraph::visit::IntoNodeIdentifiers;

    #[test]
    fn single_pattern_match_simple() {
        let g = graph();
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = SinglePatternMatcher::new(p.into_unweighted_pattern());

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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = SinglePatternMatcher::new(p.into_unweighted_pattern());
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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = SinglePatternMatcher::new(p.into_unweighted_pattern());

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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = SinglePatternMatcher::new(p.into_unweighted_pattern());

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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&pattern)
            .try_into()
            .unwrap();
        let matcher = SinglePatternMatcher::new(p.into_unweighted_pattern());

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
