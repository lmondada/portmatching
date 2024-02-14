use std::fmt;

use crate::{
    automaton::{LineBuilder, ScopeAutomaton},
    matcher::PatternMatch,
    pattern, HashMap, Pattern, PatternID, SinglePatternMatcher,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<P: Pattern> {
    automaton: ScopeAutomaton<pattern::Constraint<P>>,
    patterns: Vec<P>,
}

impl<P: Pattern> fmt::Debug for ManyMatcher<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<P: Pattern + Clone> ManyMatcher<P> {
    pub fn new(automaton: ScopeAutomaton<pattern::Constraint<P>>, patterns: Vec<P>) -> Self {
        Self {
            automaton,
            patterns,
        }
    }

    pub fn run(
        &self,
        root: pattern::Value<P>,
        graph: pattern::DataRef<P>,
    ) -> Vec<PatternMatch<PatternID, pattern::Value<P>>> {
        self.automaton
            .run(root, graph)
            .map(|id| PatternMatch::new(id, root))
            .collect()
    }

    pub fn get_match_map(
        &self,
        m: PatternMatch<PatternID, pattern::Value<P>>,
        graph: pattern::DataRef<P>,
    ) -> Option<HashMap<P::Universe, pattern::Value<P>>> {
        let p = self.patterns.get(m.pattern.0).unwrap();
        let single_matcher = SinglePatternMatcher::new(p.clone());
        single_matcher
            .get_match_map(m.root, graph)
            .map(|m| m.into_iter().collect())
    }

    pub fn get_pattern(&self, id: PatternID) -> Option<&P> {
        self.patterns.get(id.0)
    }

    pub fn from_patterns(patterns: Vec<P>) -> Self {
        let builder: LineBuilder<_> = patterns.iter().collect();
        let automaton = builder.build();
        Self::new(automaton, patterns)
    }

    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String
    where
        P::Constraint: fmt::Debug,
        pattern::Symbol<P>: fmt::Debug,
    {
        self.automaton.dot_string()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use itertools::Itertools;
    use portgraph::{
        proptest::gen_portgraph, LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView,
        UnmanagedDenseMap,
    };

    use proptest::prelude::*;

    #[cfg(feature = "serde")]
    use glob::glob;
    #[cfg(feature = "serde")]
    use std::fs;

    use crate::portgraph::{PortMatcher, PortgraphPattern, PortgraphPatternBuilder, WeightedPortGraphRef};
    use crate::{
        matcher::{ManyMatcher, PatternMatch, SinglePatternMatcher},
        utils::test::gen_portgraph_connected,
        HashSet, NaiveManyMatcher, Pattern,
    };

    use petgraph::visit::{GraphBase, IntoNodeIdentifiers};

    const DBG_DUMP_FILES: bool = false;

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = ManyMatcher::from_patterns(vec![p.into_unweighted_pattern()]);
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
                pattern: 0.into(),
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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let matcher = ManyMatcher::from_patterns(vec![p.into_unweighted_pattern()]);

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                pattern: 0.into(),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = PortgraphPattern::try_from_portgraph(&g).unwrap();
        let matcher: ManyMatcher<_> = vec![p.into_unweighted_pattern()].into();

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
        let p: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_portgraph(&g)
            .try_into()
            .unwrap();
        let _matcher: ManyMatcher<_> = vec![p].into();
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

        let p1: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_rooted_portgraph(&p1, n0)
            .try_into()
            .unwrap();
        let p2: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_rooted_portgraph(&p2, n0)
            .try_into()
            .unwrap();
        let matcher: ManyMatcher<_> =
            vec![p1.into_unweighted_pattern(), p2.into_unweighted_pattern()].into();
        assert_eq!(matcher.find_matches(&g).len(), 3);
    }

    #[test]
    fn two_simple_patterns_change_root() {
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

        let p1: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_rooted_portgraph(&p1, n0)
            .try_into()
            .unwrap();
        let p2: PortgraphPattern<_, _, _> = PortgraphPatternBuilder::from_rooted_portgraph(&p2, n0)
            .try_into()
            .unwrap();
        let matcher: ManyMatcher<_> =
            vec![p1.into_unweighted_pattern(), p2.into_unweighted_pattern()].into();
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
        ManyMatcher::from_patterns(
            [&p1, &p2]
                .map(|g| PortgraphPattern::try_from_portgraph(g).unwrap())
                .to_vec(),
        );
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

        let p1 = PortgraphPattern::try_from_portgraph(&p1).unwrap();
        let p2 = PortgraphPattern::try_from_portgraph(&p2).unwrap();
        let matcher: ManyMatcher<_> =
            vec![p1.into_unweighted_pattern(), p2.into_unweighted_pattern()].into();
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

        let p1 = PortgraphPattern::try_from_portgraph(&p1).unwrap();
        let p2 = PortgraphPattern::try_from_portgraph(&p2).unwrap();
        let _matcher: ManyMatcher<_> = vec![p1, p2].into();
    }

    // TODO: implement weighted matching
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
        let p1 = PortgraphPattern::try_from_weighted_portgraph(&p1, w1).unwrap();
        let p2 = PortgraphPattern::try_from_weighted_portgraph(&p2, w2).unwrap();
        let matcher: ManyMatcher<_> = vec![p1, p2].into();
        let mut g = PortGraph::new();
        let mut w = UnmanagedDenseMap::new();
        let n0 = g.add_node(0, 2);
        let n1 = g.add_node(2, 1);
        let n2 = g.add_node(2, 1);
        let n3 = g.add_node(1, 0);
        w[n0] = 4;
        w[n1] = 3;
        w[n2] = 2;
        w[n3] = 1;
        link(&mut g, (n0, 0), (n1, 1));
        link(&mut g, (n0, 1), (n1, 0));
        link(&mut g, (n1, 0), (n2, 0));
        link(&mut g, (n2, 0), (n3, 0));
        assert_eq!(matcher.find_matches(WeightedPortGraphRef(&g, &w)).len(), 2);
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = PortgraphPattern::try_from_portgraph(&pattern).unwrap();
            let matcher = ManyMatcher::from_patterns(vec![pattern.clone()]);
            let single_matcher = SinglePatternMatcher::new(pattern);
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
            pattern_graphs in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..100),
            g in gen_portgraph(30, 4, 60)
        ) {
            #[cfg(not(feature = "serde"))]
            if DBG_DUMP_FILES {
                println!("Warning: serde feature not enabled, cannot dump files");
            }
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                for path in glob("pattern_*.json").expect("glob pattern failed").flatten() {
                    fs::remove_file(path).expect("Removing file failed");
                }
                fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
            }
            let patterns = pattern_graphs
                .iter()
                .map(|g| PortgraphPattern::try_from_portgraph(g).unwrap())
                .collect_vec();
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                for ((i, g), p) in pattern_graphs.iter().enumerate().zip(&patterns) {
                    fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(&(g, p.root().unwrap())).unwrap()).unwrap();
                }
            }
            let naive = NaiveManyMatcher::from_patterns(patterns.clone());
            let single_matches: HashSet<_>  = naive.find_matches(&g).into_iter().collect();
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
            }
            let matcher = ManyMatcher::from_patterns(patterns);
            let many_matches: HashSet<_> = matcher.find_matches(&g).into_iter().collect();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "a bit slow"]
        #[test]
        fn many_graphs_proptest_small(
            pattern_graphs in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..10),
            g in gen_portgraph(30, 4, 60)
        ) {
            if DBG_DUMP_FILES {
                println!("Warning: serde feature not enabled, cannot dump files");
            }
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                for path in glob("pattern_*.json").expect("glob pattern failed").flatten() {
                    fs::remove_file(path).expect("Removing file failed");
                }
                fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
            }
            let patterns = pattern_graphs
                .iter()
                .map(|g| PortgraphPattern::try_from_portgraph(g).unwrap())
                .collect_vec();
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                for ((i, g), p) in pattern_graphs.iter().enumerate().zip(&patterns) {
                    fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(&(g, p.root().unwrap())).unwrap()).unwrap();
                }
            }
            let naive = NaiveManyMatcher::from_patterns(patterns.clone());
            let single_matches: BTreeSet<_> = naive.find_matches(&g).into_iter().collect();
            #[cfg(feature = "serde")]
            if DBG_DUMP_FILES {
                fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
            }
            let matcher = ManyMatcher::from_patterns(patterns);
            let many_matches: BTreeSet<_> = matcher.find_matches(&g).into_iter().collect();
            prop_assert_eq!(many_matches, single_matches);
        }
    }
}
