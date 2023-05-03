use std::collections::{BTreeMap, BTreeSet};

use portgraph::{dot::dot_string_weighted, NodeIndex, PortGraph, PortOffset};

use crate::{
    addressing::{
        cache::{Cache, SpineID},
        pg::AsPathOffset,
        AsSpineID, Skeleton, SkeletonAddressing, SpineAddress,
    },
    matcher::Matcher,
    pattern::{Edge, Pattern},
};

use super::{ManyPatternMatcher, PatternID, PatternMatch};
use crate::graph_tries::{root_state, BaseGraphTrie, GraphTrie, StateID};

/// A graph trie matcher based on skeleton partitioning of graphs.
///
/// There is some freedom in how a graph trie is built from patterns.
/// This matcher is based on partitions of the graph patterns
/// into skeleton paths.
///
/// This spreads out the occurence of non-deterministic (expensive) states in the trie
/// in-between deterministic (cheap) ones.
pub struct LineGraphTrie<T> {
    trie: T,
    match_states: BTreeMap<StateID, Vec<PatternID>>,
    patterns: Vec<Pattern>,
}

impl<T: Default> Default for LineGraphTrie<T> {
    fn default() -> Self {
        Self {
            trie: T::default(),
            match_states: Default::default(),
            patterns: Default::default(),
        }
    }
}

impl<T: Default> LineGraphTrie<T> {
    /// Create a new empty matcher.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: Clone> LineGraphTrie<BaseGraphTrie<S>> {
    /// A dotstring representation of the trie.
    pub fn dotstring(&self) -> String {
        let mut weights = self.trie.str_weights();
        for n in self.trie.graph.nodes_iter() {
            let empty = vec![];
            let matches = self.match_states.get(&n).unwrap_or(&empty);
            if !matches.is_empty() {
                weights[n] += &format!("[{:?}]", matches);
            }
        }
        dot_string_weighted(&self.trie.graph, &weights)
    }
}

impl LineGraphTrie<BaseGraphTrie<(Vec<PortOffset>, usize)>> {
    /// Convert the trie to using [`SpineID`]s for caching.
    pub fn to_cached_trie(
        &self,
    ) -> LineGraphTrie<BaseGraphTrie<(SpineID, Vec<PortOffset>, usize)>> {
        LineGraphTrie {
            trie: self.trie.to_cached_trie(),
            match_states: self.match_states.clone(),
            patterns: self.patterns.clone(),
        }
    }
}

impl<T> Matcher for LineGraphTrie<T>
where
    T: GraphTrie,
    for<'n> <<T as GraphTrie>::SpineID as SpineAddress>::AsRef<'n>:
        Copy + AsSpineID + AsPathOffset + PartialEq,
{
    type Match = PatternMatch;

    fn find_anchored_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Self::Match> {
        let mut current_states = vec![root_state()];
        let mut matches = BTreeSet::new();
        let addressing = T::Addressing::init(root, graph);
        let mut cache = Cache::default();
        while !current_states.is_empty() {
            let mut new_states = Vec::new();
            for state in current_states {
                for &pattern_id in self.match_states.get(&state).into_iter().flatten() {
                    matches.insert(PatternMatch {
                        id: pattern_id,
                        root,
                    });
                }
                for next_state in self.trie.next_states(state, &addressing, &mut cache) {
                    new_states.push(next_state);
                }
            }
            current_states = new_states;
        }
        Vec::from_iter(matches)
    }
}

impl ManyPatternMatcher for LineGraphTrie<BaseGraphTrie<(Vec<PortOffset>, usize)>> {
    fn add_pattern(&mut self, pattern: Pattern) -> PatternID {
        // The pattern number of this pattern
        let pattern_id = PatternID(self.patterns.len());
        self.patterns.push(pattern);
        let pattern = &self.patterns[pattern_id.0];
        let graph = &pattern.graph;
        let skeleton = Skeleton::new(graph, pattern.root);

        // Stores the current positions in the graph trie, along with the
        // match that corresponds to that position
        let mut current_states = [root_state()].into();

        // Decompose a pattern into "lines", which are paths in the pattern
        let all_lines = pattern.all_lines();

        for line in all_lines {
            // Traverse the line
            let mut first_edge = true;
            for Edge(out_port, _) in line {
                if first_edge {
                    // The edge is added non-deterministically
                    current_states = self.trie.add_graph_edge_nondet(
                        out_port,
                        current_states,
                        &skeleton,
                        // &mut clone_state,
                    );
                } else {
                    // All other edges are deterministic
                    current_states = self.trie.add_graph_edge_det(
                        out_port,
                        current_states,
                        &skeleton,
                        // &mut clone_state,
                    );
                }
                first_edge = false;
            }
        }

        // A callback when a state is cloned in the trie
        // necessary to keep track of the match states
        let clone_state = |old_state: StateID, new_state: StateID| {
            self.match_states.insert(
                new_state,
                self.match_states
                    .get(&old_state)
                    .cloned()
                    .unwrap_or_default(),
            );
        };

        self.trie.finalize(clone_state);

        // Record matching pattern in final states
        for state in current_states {
            self.match_states.entry(state).or_default().push(pattern_id);
        }

        pattern_id
    }
}

#[cfg(test)]
mod tests {
    // use std::fs;
    // use glob::glob;

    #[cfg(feature = "serde")]
    use itertools::Itertools;

    use portgraph::{proptest::gen_portgraph, NodeIndex, PortGraph, PortOffset};

    use proptest::prelude::*;

    use crate::{
        graph_tries::BaseGraphTrie,
        matcher::{
            many_patterns::{
                line_based::LineGraphTrie, ManyPatternMatcher, PatternID, PatternMatch,
            },
            Matcher, SinglePatternMatcher,
        },
        pattern::Pattern,
        utils::test_utils::gen_portgraph_connected,
    };

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = LineGraphTrie::<BaseGraphTrie>::new();
        matcher.add_pattern(p);
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
                id: PatternID(0),
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
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(p);

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                id: PatternID(0),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(p);

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

    #[test]
    fn single_pattern_construction() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(3, 2);
        let n1 = g.add_node(2, 0);
        let n2 = g.add_node(2, 1);
        link(&mut g, (n0, 0), (n0, 2));
        link(&mut g, (n0, 1), (n1, 1));
        link(&mut g, (n2, 0), (n0, 1));
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(p);
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

        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
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
        LineGraphTrie::from_patterns([p1, p2].map(|p| Pattern::from_graph(p).unwrap()));
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

        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
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

        let mut matcher = LineGraphTrie::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_graph(pattern).unwrap();
            let mut matcher = LineGraphTrie::new();
            let pattern_id = matcher.add_pattern(pattern.clone());
            let single_matcher = SinglePatternMatcher::from_pattern(pattern.clone());
            let many_matches = matcher.find_matches(&g);
            let single_matches: Vec<_> = single_matcher
                .find_matches(&g)
                .into_iter()
                .map(|m| PatternMatch {
                    id: pattern_id,
                    root: m[&pattern.root],
                })
                .collect();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "a bit slow"]
        #[cfg(feature = "serde")]
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
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            let single_matchers = patterns
                .clone()
                .into_iter()
                .map(SinglePatternMatcher::from_pattern)
                .collect_vec();
            let single_matches = single_matchers
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    m.find_matches(&g)
                        .into_iter()
                        .map(|m| PatternMatch {
                            id: PatternID(i),
                            root: m[&patterns[i].root],
                        })
                        .collect_vec()
                })
                .collect_vec();
            // fs::write("results.bin", rmp_serde::to_vec(&single_matches).unwrap()).unwrap();
            let matcher = LineGraphTrie::from_patterns(patterns.clone());
            let many_matches = matcher.find_matches(&g);
            let many_matches = (0..patterns.len())
                .map(|i| {
                    many_matches
                        .iter()
                        .filter(|m| m.id == PatternID(i))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "a bit slow"]
        #[cfg(feature = "serde")]
        #[test]
        fn many_graphs_proptest_no_cached(
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
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            let single_matchers = patterns
                .clone()
                .into_iter()
                .map(SinglePatternMatcher::from_pattern)
                .collect_vec();
            let single_matches = single_matchers
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    m.find_matches(&g)
                        .into_iter()
                        .map(|m| PatternMatch {
                            id: PatternID(i),
                            root: m[&patterns[i].root],
                        })
                        .collect_vec()
                })
                .collect_vec();
            // fs::write("results.bin", rmp_serde::to_vec(&single_matches).unwrap()).unwrap();
            let matcher = LineGraphTrie::from_patterns(patterns.clone()).to_cached_trie();
            let many_matches = matcher.find_matches(&g);
            let many_matches = (0..patterns.len())
                .map(|i| {
                    many_matches
                        .iter()
                        .filter(|m| m.id == PatternID(i))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    // #[test]
    // fn traverse_from_test() {
    //     let mut matcher = ManyPatternMatcher::new();
    //     let tree = &mut matcher.tree.line_trees[0];
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(0)),
    //         TreeNodeID::SameTree(1),
    //     );
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(1)),
    //         TreeNodeID::SameTree(2),
    //     );
    //     tree.push(TreeNode {
    //         out_port: Some(PortOffset(1)),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(
    //             NodeTransition::NewNode(PortOffset(0)),
    //             TreeNodeID::SameTree(3),
    //         )]
    //         .into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(2).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(1).into(),
    //         address: PatternNodeAddress(0, 2),
    //         transitions: [].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(0).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v3 = g.add_node(1, 1);
    //     let v4 = g.add_node(1, 0);
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v1_0 = g.port_index(v1, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v0_0, v1_0).unwrap();
    //     let v1_1 = g.port_index(v1, 0, Direction::Outgoing).unwrap();
    //     let v3_0 = g.port_index(v3, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v1_1, v3_0).unwrap();
    //     let v3_1 = g.port_index(v3, 0, Direction::Outgoing).unwrap();
    //     let v4_0 = g.port_index(v4, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v3_1, v4_0).unwrap();
    //     let line = vec![
    //         Edge(v0_0, v1_0.into()),
    //         Edge(v1_1, v3_0.into()),
    //         Edge(v3_1, v4_0.into()),
    //     ];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line),
    //         (3, Some(&Edge(v3_1, v4_0.into())))
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v1, PatternNodeAddress(0, 1)),
    //             (v3, PatternNodeAddress(0, 2)),
    //         ]
    //         .into()
    //     );

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v2_1 = g.port_index(v2, 1, Direction::Incoming).unwrap();
    //     let v2_2 = g.port_index(v2, 0, Direction::Outgoing).unwrap();
    //     g.link_ports(v0_0, v2_1).unwrap();
    //     let line = vec![Edge(v0_0, v2_1.into()), Edge(v2_2, None)];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line, TransitionPolicy::NoDanglingEdge),
    //         (4, None)
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v2, PatternNodeAddress(0, 1)),
    //         ]
    //         .into()
    //     );
    // }
}
