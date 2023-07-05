use std::{borrow::Borrow, fmt, hash::Hash};

use itertools::Itertools;
use portgraph::{LinkView, NodeIndex, PortGraph, PortOffset, PortView};

use crate::{
    automaton::{LineBuilder, ScopeAutomaton},
    matcher::{Match, PatternMatch},
    patterns::{compatible_offsets, UnweightedEdge},
    EdgeProperty, NodeProperty, Pattern, PortMatcher, Universe,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<U: Universe, PNode, PEdge: Eq + Hash> {
    automaton: ScopeAutomaton<PNode, PEdge>,
    patterns: Vec<Pattern<U, PNode, PEdge>>,
}

// impl<U: Universe, PNode: Property, PEdge: Property> ManyMatcher<U, PNode, PEdge> {
//     pub fn from_line_patterns(patterns: Vec<LinePattern<U, PNode, PEdge>>) -> Self {
//         let builder = LineBuilder::from_patterns(patterns);
//         let automaton = builder.build();
//         Self {
//             automaton,
//             patterns,
//         }
//     }
// }

impl<U: Universe, PNode: NodeProperty> ManyMatcher<U, PNode, UnweightedEdge> {
    pub fn from_patterns(patterns: Vec<Pattern<U, PNode, UnweightedEdge>>) -> Self {
        let line_patterns = patterns
            .clone()
            .into_iter()
            .map(|p| {
                p.try_into_line_pattern(compatible_offsets)
                    .expect("Failed to express pattern as line pattern")
            })
            .collect_vec();
        let builder = LineBuilder::from_patterns(line_patterns);
        let automaton = builder.build();
        Self {
            automaton,
            patterns,
        }
    }
}

impl<U: Universe, PNode: Copy + fmt::Debug, PEdge: Copy + Eq + Hash + fmt::Debug>
    ManyMatcher<U, PNode, PEdge>
{
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}

pub type UnweightedManyMatcher = ManyMatcher<NodeIndex, (), UnweightedEdge>;

impl<'g, G> PortMatcher<G> for ManyMatcher<NodeIndex, (), UnweightedEdge>
where
    G: Borrow<PortGraph>,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: G, root: NodeIndex) -> Vec<Match<'_, Self, G>> {
        // Node weights (none)
        let node_prop = |_, ()| true;
        // Check edges exist
        let edge_prop = |n, (pout, pin)| {
            let graph = graph.borrow();
            let out_port = graph.port_index(n, pout)?;
            let in_port = graph.port_link(out_port)?;
            if graph.port_offset(out_port).unwrap() != pout
                || graph.port_offset(in_port).unwrap() != pin
            {
                return None;
            }
            graph.port_node(in_port)
        };
        self.automaton
            .run(root, node_prop, edge_prop)
            .map(|p_id| PatternMatch {
                pattern: &self.patterns[p_id.0],
                root,
            })
            .collect()
    }
}

impl<U: Universe, PNode: NodeProperty> From<Vec<Pattern<U, PNode, UnweightedEdge>>>
    for ManyMatcher<U, PNode, UnweightedEdge>
{
    fn from(value: Vec<Pattern<U, PNode, UnweightedEdge>>) -> Self {
        Self::from_patterns(value)
    }
}

#[cfg(test)]
mod tests {
    // use glob::glob;
    // use std::fs;

    use itertools::Itertools;

    use portgraph::{
        proptest::gen_portgraph, LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView,
        UnmanagedDenseMap,
    };

    use proptest::prelude::*;

    use crate::{
        matcher::{ManyMatcher, PatternMatch, PortMatcher, SinglePatternMatcher},
        utils::test::gen_portgraph_connected,
        NaiveManyMatcher, Pattern,
    };

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_portgraph(&g);
        let matcher = ManyMatcher::from_patterns(vec![p.clone()]);
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                pattern: &p,
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_loop_link2() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_portgraph(&g);
        let matcher = ManyMatcher::from_patterns(vec![p.clone()]);

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                pattern: &p,
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = Pattern::from_portgraph(&g);
        let matcher: ManyMatcher<_, _, _> = vec![p].into();

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n1, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        // We no longer match dangling ports
        // assert_eq!(matcher.find_matches(&g), vec![]);
        assert_eq!(matcher.find_matches(&g).len(), 2);
    }

    #[test]
    fn single_pattern_construction() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(3, 2);
        let n1 = g.add_node(2, 0);
        let n2 = g.add_node(2, 1);
        link(&mut g, (n0, 0), (n0, 2));
        link(&mut g, (n0, 1), (n1, 1));
        link(&mut g, (n2, 0), (n0, 1));
        let p = Pattern::from_portgraph(&g);
        let matcher: ManyMatcher<_, _, _> = vec![p].into();
    }

    fn link(p: &mut PortGraph, (n1, p1): (NodeIndex, usize), (n2, p2): (NodeIndex, usize)) {
        p.link_ports(
            p.port_index(n1, PortOffset::new_outgoing(p1)).unwrap(),
            p.port_index(n2, PortOffset::new_incoming(p2)).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn two_simple_patterns() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(0, 1);
        let n1 = p1.add_node(2, 0);
        let n2 = p1.add_node(0, 1);
        link(&mut p1, (n0, 0), (n1, 0));
        link(&mut p1, (n2, 0), (n1, 1));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 2);
        let n1 = g.add_node(3, 1);
        let n2 = g.add_node(0, 2);
        let n3 = g.add_node(3, 2);
        link(&mut g, (n0, 0), (n1, 2));
        link(&mut g, (n0, 1), (n3, 0));
        link(&mut g, (n1, 0), (n3, 1));
        link(&mut g, (n2, 0), (n1, 0));
        link(&mut g, (n2, 1), (n3, 2));
        link(&mut g, (n3, 0), (n1, 1));
        link(&mut g, (n3, 1), (n0, 0));

        let p1 = Pattern::from_portgraph(&p1);
        let p2 = Pattern::from_portgraph(&p2);
        let matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();

        assert_eq!(matcher.find_matches(&g).len(), 3);
    }

    #[test]
    fn trie_construction_fail() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(2, 1);
        let n1 = p1.add_node(1, 0);
        link(&mut p1, (n0, 0), (n1, 0));
        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 0);
        let n1 = p2.add_node(0, 2);
        link(&mut p2, (n1, 0), (n0, 1));
        link(&mut p2, (n1, 1), (n0, 0));
        ManyMatcher::from_patterns([&p1, &p2].map(Pattern::from_portgraph).to_vec());
    }

    #[test]
    fn two_simple_patterns_2() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(1, 0);
        let n1 = p1.add_node(1, 3);
        let n2 = p1.add_node(0, 3);
        let n3 = p1.add_node(1, 3);
        link(&mut p1, (n1, 0), (n0, 0));
        link(&mut p1, (n2, 0), (n1, 0));
        link(&mut p1, (n2, 2), (n3, 0));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let p1 = Pattern::from_portgraph(&p1);
        let p2 = Pattern::from_portgraph(&p2);
        let matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();
        assert_eq!(matcher.find_matches(&g).len(), 1);
    }
    #[test]
    fn two_simple_patterns_3() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(3, 1);
        let n1 = p1.add_node(0, 3);
        link(&mut p1, (n1, 1), (n0, 0));
        link(&mut p1, (n0, 0), (n0, 2));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(0, 2);
        let n1 = p2.add_node(2, 0);
        link(&mut p2, (n0, 0), (n1, 1));
        link(&mut p2, (n0, 1), (n1, 0));

        let p1 = Pattern::from_portgraph(&p1);
        let p2 = Pattern::from_portgraph(&p2);
        let matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();
    }

    #[test]
    fn weighted_pattern_matching() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(2, 1);
        let n1 = p1.add_node(1, 0);
        link(&mut p1, (n0, 0), (n1, 0));
        let mut w1 = UnmanagedDenseMap::new();
        w1[n0] = 2;
        w1[n1] = 1;
        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 0);
        let n1 = p2.add_node(0, 2);
        link(&mut p2, (n1, 0), (n0, 1));
        link(&mut p2, (n1, 1), (n0, 0));
        let mut w2 = UnmanagedDenseMap::new();
        w2[n0] = 3;
        w2[n1] = 4;
        todo!();
        // let p1 = Pattern::from_weighted_portgraph(&p1, w1);
        // let p2 = Pattern::from_weighted_portgraph(&p2, w2);
        // let matcher = vec![p1, p2].into();
        // let mut g = PortGraph::new();
        // let mut w = UnmanagedDenseMap::new();
        // let n0 = g.add_node(0, 2);
        // let n1 = g.add_node(2, 1);
        // let n2 = g.add_node(2, 1);
        // let n3 = g.add_node(1, 0);
        // w[n0] = 4;
        // w[n1] = 3;
        // w[n2] = 2;
        // w[n3] = 1;
        // link(&mut g, (n0, 0), (n1, 1));
        // link(&mut g, (n0, 1), (n1, 0));
        // link(&mut g, (n1, 0), (n2, 0));
        // link(&mut g, (n2, 0), (n3, 0));
        // assert_eq!(matcher.find_weighted_matches(&g, &w).len(), 2);
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_portgraph(&pattern);
            let matcher = ManyMatcher::from_patterns(vec![pattern.clone()]);
            let single_matcher = SinglePatternMatcher::from_pattern(pattern.clone());
            let many_matches = matcher.find_matches(&g);
            let single_matches: Vec<_> = single_matcher
                .find_matches(&g);
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "a bit slow"]
        #[test]
        fn many_graphs_proptest(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..100),
            g in gen_portgraph(30, 4, 60)
        ) {
            // for entry in glob("pattern_*.bin").expect("glob pattern failed") {
            //     match entry {
            //         Ok(path) => fs::remove_file(path).expect("Removing file failed"),
            //         Err(_) => {},
            //     }
            // }
            // for (i, p) in patterns.iter().enumerate() {
            //     fs::write(&format!("pattern_{}.bin", i), rmp_serde::to_vec(p).unwrap()).unwrap();
            // }
            // fs::write("graph.bin", rmp_serde::to_vec(&g).unwrap()).unwrap();
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_portgraph(&p))
                .collect_vec();
            let naive = NaiveManyMatcher::from_patterns(patterns.clone());
            let single_matches = naive.find_matches(&g);
            // fs::write("results.bin", rmp_serde::to_vec(&single_matches).unwrap()).unwrap();
            let matcher = ManyMatcher::from_patterns(patterns.clone());
            let many_matches = matcher.find_matches(&g);
            // TODO: order should not matter
            // prop_assert_eq!(BTreeSet::from_iter(many_matches), BTreeSet::from_iter(single_matches));
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "a bit slow"]
        #[test]
        fn many_graphs_proptest_small(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..10),
            g in gen_portgraph(30, 4, 60)
        ) {
            // for entry in glob("pattern_*.json").expect("glob pattern failed") {
            //     if let Ok(path) = entry {
            //         fs::remove_file(path).expect("Removing file failed");
            //     }
            // }
            // for (i, p) in patterns.iter().enumerate() {
            //     fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(p).unwrap()).unwrap();
            // }
            // fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_portgraph(&p))
                .collect_vec();
            let naive = NaiveManyMatcher::from_patterns(patterns.clone());
            let single_matches = naive.find_matches(&g);
            // fs::write("results.bin", rmp_serde::to_vec(&single_matches).unwrap()).unwrap();
            let matcher = ManyMatcher::from_patterns(patterns.clone());
            let many_matches = matcher.find_matches(&g);
            // TODO: order should not matter
            // prop_assert_eq!(BTreeSet::from_iter(many_matches), BTreeSet::from_iter(single_matches));
            prop_assert_eq!(many_matches, single_matches);
        }
    }
}
