use std::fmt;

use itertools::Itertools;
use portgraph::{LinkView, NodeIndex, SecondaryMap};

use crate::{
    automaton::{LineBuilder, ScopeAutomaton},
    matcher::{Match, PatternMatch},
    patterns::{compatible_offsets, UnweightedEdge},
    utils::{always_true, validate_unweighted_edge, validate_weighted_node},
    EdgeProperty, HashMap, NodeProperty, Pattern, PatternID, PortMatcher, SinglePatternMatcher,
    Universe, WeightedGraphRef,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<U, PNode, PEdge, OffsetID = <PEdge as EdgeProperty>::OffsetID>
where
    PEdge: EdgeProperty,
    U: Universe,
{
    automaton: ScopeAutomaton<PNode, PEdge, OffsetID>,
    patterns: Vec<Pattern<U, PNode, PEdge>>,
}

impl<U: Universe, PNode, PEdge: EdgeProperty> fmt::Debug for ManyMatcher<U, PNode, PEdge> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty> ManyMatcher<U, PNode, PEdge> {
    pub fn new(
        automaton: ScopeAutomaton<PNode, PEdge, <PEdge as EdgeProperty>::OffsetID>,
        patterns: Vec<Pattern<U, PNode, PEdge>>,
    ) -> Self {
        Self {
            automaton,
            patterns,
        }
    }

    pub fn run<N: Universe>(
        &self,
        root: N,
        validate_node: impl for<'a> Fn(N, &PNode) -> bool,
        validate_edge: impl for<'a> Fn(N, &PEdge) -> Option<N>,
    ) -> Vec<PatternMatch<PatternID, N>> {
        self.automaton
            .run(root, validate_node, validate_edge)
            .map(|id| PatternMatch::new(id, root))
            .collect()
    }

    pub fn get_match_map<N: Universe>(
        &self,
        m: PatternMatch<PatternID, N>,
        validate_node: impl for<'a> Fn(N, &PNode) -> bool,
        validate_edge: impl for<'a> Fn(N, &'a PEdge) -> Option<N>,
    ) -> Option<HashMap<U, N>> {
        let p = self.patterns.get(m.pattern.0).unwrap();
        let single_matcher = SinglePatternMatcher::from_pattern(p.clone());
        single_matcher
            .get_match_map(m.root, validate_node, validate_edge)
            .map(|m| m.into_iter().collect())
    }

    pub fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, PNode, PEdge>> {
        self.patterns.get(id.0)
    }
}

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
        Self::new(automaton, patterns)
    }
}

impl<U, PNode, PEdge> ManyMatcher<U, PNode, PEdge>
where
    <PEdge as EdgeProperty>::OffsetID: fmt::Debug,
    U: Universe,
    PNode: Copy + fmt::Debug,
    PEdge: EdgeProperty + fmt::Debug,
{
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}

pub type UnweightedManyMatcher = ManyMatcher<NodeIndex, (), UnweightedEdge>;

impl<U, G> PortMatcher<G, NodeIndex, U> for ManyMatcher<U, (), UnweightedEdge>
where
    U: Universe,
    G: LinkView,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: G, root: NodeIndex) -> Vec<Match> {
        self.run(
            root,
            // no node prop
            always_true,
            // check edge exist
            validate_unweighted_edge(graph),
        )
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, Self::PNode, Self::PEdge>> {
        self.get_pattern(id)
    }
}

impl<U: Universe, PNode: NodeProperty> From<Vec<Pattern<U, PNode, UnweightedEdge>>>
    for ManyMatcher<U, PNode, UnweightedEdge>
{
    fn from(value: Vec<Pattern<U, PNode, UnweightedEdge>>) -> Self {
        Self::from_patterns(value)
    }
}

impl<'m, U, G, W, M> PortMatcher<WeightedGraphRef<G, &'m M>, NodeIndex, U>
    for ManyMatcher<U, W, UnweightedEdge>
where
    M: SecondaryMap<NodeIndex, W>,
    G: LinkView + Copy,
    U: Universe,
    W: NodeProperty,
{
    type PNode = W;
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(
        &self,
        weighted_graph: WeightedGraphRef<G, &'m M>,
        root: NodeIndex,
    ) -> Vec<Match> {
        let (graph, _) = weighted_graph.into();
        self.run(
            root,
            // Node weights (none)
            validate_weighted_node(weighted_graph),
            // Check edges exist
            validate_unweighted_edge(graph),
        )
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, Self::PNode, Self::PEdge>> {
        self.patterns.get(id.0)
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


    use crate::{
        matcher::{ManyMatcher, PatternMatch, PortMatcher, SinglePatternMatcher},
        utils::test::gen_portgraph_connected,
        HashSet, NaiveManyMatcher, Pattern,
    };

    const DBG_DUMP_FILES: bool = false;

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_portgraph(&g);
        let matcher = ManyMatcher::from_patterns(vec![p]);
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
        let p = Pattern::from_portgraph(&g);
        let matcher = ManyMatcher::from_patterns(vec![p]);

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
        let _matcher: ManyMatcher<_, _, _> = vec![p].into();
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

        let p1 = Pattern::from_rooted_portgraph(&p1, n0);
        let p2 = Pattern::from_rooted_portgraph(&p2, n0);
        let matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();
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

        let p1 = Pattern::from_rooted_portgraph(&p1, n1);
        let p2 = Pattern::from_rooted_portgraph(&p2, n0);
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
        let _matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();
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
        let p1 = Pattern::from_weighted_portgraph(&p1, w1);
        let p2 = Pattern::from_weighted_portgraph(&p2, w2);
        let matcher: ManyMatcher<_, _, _> = vec![p1, p2].into();
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
        assert_eq!(matcher.find_matches((&g, &w).into()).len(), 2);
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_portgraph(&pattern);
            let matcher = ManyMatcher::from_patterns(vec![pattern.clone()]);
            let single_matcher = SinglePatternMatcher::from_pattern(pattern);
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
                .map(Pattern::from_portgraph)
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
                .map(Pattern::from_portgraph)
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
