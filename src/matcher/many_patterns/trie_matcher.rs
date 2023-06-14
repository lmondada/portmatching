use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
};

use itertools::Itertools;
use portgraph::{
    algorithms::postorder, dot::dot_string_weighted, Direction, NodeIndex, PortGraph, PortIndex,
};

use crate::{
    constraint::{mutually_exclusive, totally_ordered, Address, ConstraintType, PortAddress},
    graph_tries::{root_state, BaseGraphTrie, GraphTrie, StateID},
    pattern::{Edge, Pattern},
    Constraint, ManyPatternMatcher, Matcher, PatternID, Skeleton,
};

use super::PatternMatch;

/// A graph pattern matcher using graph tries.
///
/// There is some freedom in how a graph trie is built from patterns.
/// We support three different strategies, with various performance tradeoffs:
///  * [`Deterministic`]: Use only deterministic states. In theory, this should
///    perform best, but trie sizes grow exponentially, so not recommended in practice.
///  * [`NonDeterministic`]: Use only non-deterministic states. This should perform
///    very similarly to pattern matching each
///    pattern independently, as there is no "smartness" in the trie.
///  * [`Balanced`]: Based on partitions of the graph patterns into skeleton
///    paths, alternate between deterministic and non-deterministic states. For
///    circuit-like graphs the number of non-deterministic states is bounded.
///
/// [`Deterministic`]: TrieConstruction::Deterministic
/// [`NonDeterministic`]: TrieConstruction::NonDeterministic
/// [`Balanced`]: TrieConstruction::Balanced
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrieMatcher<C, A, P> {
    strategy: TrieConstruction,
    trie: BaseGraphTrie<C, A>,
    match_states: BTreeMap<StateID, Vec<PatternID>>,
    patterns: Vec<P>,
}

/// Trie construction strategy.
///
/// See [`TrieMatcher`] for details on the different strategies.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrieConstruction {
    /// Use only deterministic states.
    Deterministic,
    /// Use only non-deterministic states.
    NonDeterministic,
    /// Use a mix of deterministic and non-deterministic states.
    Balanced,
}

impl<C: Clone + Ord + Constraint, A: Clone + Ord, P> Default for TrieMatcher<C, A, P> {
    fn default() -> Self {
        Self::new(TrieConstruction::Balanced)
    }
}

impl<C: Constraint + Clone + Ord, A: Clone + Ord, P> TrieMatcher<C, A, P> {
    /// Create a new matcher with the given trie construction strategy.
    pub fn new(strategy: TrieConstruction) -> Self {
        Self {
            strategy,
            trie: Default::default(),
            match_states: Default::default(),
            patterns: Default::default(),
        }
    }

    fn add_non_det(&self, is_first_edge: bool) -> bool {
        match self.strategy {
            TrieConstruction::Deterministic => false,
            TrieConstruction::NonDeterministic => true,
            TrieConstruction::Balanced => is_first_edge,
        }
    }
}

impl<C: Clone + Ord + Constraint, P> TrieMatcher<C, Address, P>
where
    P: Pattern<Constraint = C>,
{
    /// Spread transitions across nodes to minimise the number of constraints
    /// to check
    pub fn optimise(&mut self, cutoff: usize, depth_cutoff: usize)
    where
        C: ConstraintType,
        C::CT: Ord,
    {
        let mut has_changed = true;
        while has_changed {
            has_changed = false;
            for node in self.trie.large_non_det_states(cutoff, depth_cutoff) {
                // Split `node` into a tree of mostly deterministic elementary nodes
                let bottom_nodes = self.trie.split_into_det_tree(node);

                // now make bottom nodes deterministic
                for bottom_node in bottom_nodes.into_iter().unique() {
                    has_changed |= self.try_make_det(bottom_node);
                }
                if has_changed {
                    break;
                }
            }
        }
    }

    /// Try turning `node` into deterministc node
    fn try_make_det(&mut self, node: NodeIndex) -> bool {
        let constraints = self.trie.all_constraints(node);
        let is_totally_ordered = totally_ordered(&constraints);
        for i in (0..self.trie.graph.num_outputs(node)).rev() {
            let Some(next_p) = self.trie.graph.output(node, i + 1) else {
                continue
            };
            let p = self.trie.graph.output(node, i).unwrap();
            if is_totally_ordered || self.trie.transition(p) == self.trie.transition(next_p) {
                self.merge_ports(p, next_p);
            }
        }
        self.trie.remove_duplicate_ports(node);
        let constraints = self.trie.all_constraints(node);
        if is_totally_ordered || mutually_exclusive(&constraints) {
            self.trie.weights[node].non_deterministic = false;
            true
        } else {
            false
        }
    }

    fn merge_ports(&mut self, into: PortIndex, other: PortIndex) {
        let child = |p| {
            let link = self.trie.graph.port_link(p)?;
            self.trie.graph.port_node(link)
        };
        let other_state = child(other).expect("unlinked port");
        let into_state = child(into).expect("unlinked port");
        // List all patterns that can be reached
        let other_patterns = postorder(&self.trie.graph, [other_state], Direction::Outgoing)
            .flat_map(|n| self.match_states.get(&n).into_iter().flatten().copied())
            .collect::<BTreeSet<_>>();
        let into_patterns = postorder(&self.trie.graph, [into_state], Direction::Outgoing)
            .flat_map(|n| self.match_states.get(&n).into_iter().flatten().copied())
            .collect::<BTreeSet<_>>();
        // Re-Link into port with other port
        for removed_node in self.trie.relink(into, other_state) {
            self.match_states.remove(&removed_node);
        }

        // Now add all into patterns not in other
        // TODO: maybe other \ into has less patterns, we could merge the other way around
        for &pattern_id in into_patterns.difference(&other_patterns) {
            // TODO: find out how much of the pattern must be added

            self.trie.reset();

            let all_lines = self.patterns[pattern_id.0].all_lines();
            self.trie.insert_trace(into);
            let next = self.trie.graph.port_link(into).expect("unlinked port");
            let next = self.trie.graph.port_node(next).expect("invalid port");
            self.add_lines(&all_lines, pattern_id, next);
            // manually add trace

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

            let root = self.trie.graph.port_node(into).expect("invalid port");
            let current_states = self.trie.finalize(root, clone_state);

            // Record matching pattern in final states
            for state in current_states {
                self.match_states.entry(state).or_default().push(pattern_id);
            }
        }
    }

    fn add_lines<'a>(
        &mut self,
        lines: impl IntoIterator<Item = impl IntoIterator<Item = &'a Edge>>,
        pattern_id: PatternID,
        root: StateID,
    ) {
        let pattern = &self.patterns[pattern_id.0];
        let skeleton = Skeleton::new(pattern.graph(), pattern.root());

        // Stores the current positions in the graph trie, along with the
        // match that corresponds to that position
        let mut current_states = [root].into();

        for line in lines {
            // Traverse the line
            let mut first_edge = true;
            for e @ &Edge(out_port, _) in line {
                let constraint = pattern.to_constraint(e);
                if self.add_non_det(first_edge) {
                    // The edge is added non-deterministically
                    current_states = self.trie.add_graph_edge_nondet(
                        &skeleton.get_port_address(out_port),
                        current_states,
                        constraint,
                    );
                } else {
                    // All other edges are deterministic
                    current_states = self.trie.add_graph_edge_det(
                        &skeleton.get_port_address(out_port),
                        current_states,
                        constraint,
                    );
                }
                first_edge = false;
            }
        }
    }
}

impl<C: Display + Clone, A: Debug + Clone, P> TrieMatcher<C, A, P> {
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

type Graph<'g> = (&'g PortGraph, NodeIndex);

impl<'g, C, A, P> Matcher<Graph<'g>> for TrieMatcher<C, A, P>
where
    C: Clone + Constraint<Graph<'g> = Graph<'g>> + Ord,
    A: Clone + Ord + PortAddress<Graph<'g>>,
{
    type Match = PatternMatch;

    fn find_anchored_matches(&self, g @ (_, root): Graph<'g>) -> Vec<Self::Match> {
        let mut current_states = vec![root_state()];
        let mut matches = BTreeSet::new();
        while !current_states.is_empty() {
            let mut new_states = Vec::new();
            for state in current_states {
                for &pattern_id in self.match_states.get(&state).into_iter().flatten() {
                    matches.insert(PatternMatch {
                        id: pattern_id,
                        root,
                    });
                }
                for next_state in self.trie.next_states(state, g) {
                    new_states.push(next_state);
                }
            }
            current_states = new_states;
        }
        Vec::from_iter(matches)
    }
}

type GraphBis<'g, W> = (&'g PortGraph, W, NodeIndex);

impl<'g, C, A, W: Copy, P> Matcher<GraphBis<'g, W>> for TrieMatcher<C, A, P>
where
    C: Clone + Constraint<Graph<'g> = GraphBis<'g, W>> + Ord,
    A: Clone + Ord + PortAddress<GraphBis<'g, W>>,
{
    type Match = PatternMatch;

    fn find_anchored_matches(&self, g @ (_, _, root): GraphBis<'g, W>) -> Vec<Self::Match> {
        let mut current_states = vec![root_state()];
        let mut matches = BTreeSet::new();
        while !current_states.is_empty() {
            let mut new_states = Vec::new();
            for state in current_states {
                for &pattern_id in self.match_states.get(&state).into_iter().flatten() {
                    matches.insert(PatternMatch {
                        id: pattern_id,
                        root,
                    });
                }
                for next_state in self.trie.next_states(state, g) {
                    new_states.push(next_state);
                }
            }
            current_states = new_states;
        }
        Vec::from_iter(matches)
    }
}

impl<C: Clone + Ord + Constraint, G, P> ManyPatternMatcher<G, P> for TrieMatcher<C, Address, P>
where
    Self: Matcher<G>,
    P: Pattern<Constraint = C>,
{
    type Constraint = C;

    fn add_pattern(&mut self, pattern: P) -> PatternID {
        // The pattern number of this pattern
        let pattern_id = PatternID(self.patterns.len());
        self.patterns.push(pattern);

        // Decompose a pattern into "lines", which are paths in the pattern
        let all_lines = self.patterns[pattern_id.0].all_lines();

        self.add_lines(&all_lines, pattern_id, root_state());

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

        let current_states = self.trie.finalize(root_state(), clone_state);

        // Record matching pattern in final states
        for state in current_states {
            self.match_states.entry(state).or_default().push(pattern_id);
        }

        pattern_id
    }
}

#[cfg(test)]
mod tests {
    // use glob::glob;
    // use std::fs;

    use itertools::Itertools;

    use portgraph::{proptest::gen_portgraph, NodeIndex, PortGraph, PortOffset, UnmanagedDenseMap};

    use proptest::prelude::*;

    use crate::{
        matcher::{
            many_patterns::{
                trie_matcher::{TrieConstruction, TrieMatcher},
                ManyPatternMatcher, PatternID, PatternMatch,
            },
            Matcher, SinglePatternMatcher,
        },
        pattern::{UnweightedPattern, WeightedPattern},
        utils::test_utils::gen_portgraph_connected,
    };

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = UnweightedPattern::from_graph(g.clone()).unwrap();
        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
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
        let p = UnweightedPattern::from_graph(g.clone()).unwrap();
        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
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
        let p = UnweightedPattern::from_graph(g).unwrap();
        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
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
        let p = UnweightedPattern::from_graph(g).unwrap();
        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
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

        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
        matcher.add_pattern(UnweightedPattern::from_graph(p1).unwrap());
        matcher.add_pattern(UnweightedPattern::from_graph(p2).unwrap());
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
        TrieMatcher::from_patterns([p1, p2].map(|p| UnweightedPattern::from_graph(p).unwrap()));
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

        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
        matcher.add_pattern(UnweightedPattern::from_graph(p1).unwrap());
        matcher.add_pattern(UnweightedPattern::from_graph(p2).unwrap());
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

        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
        matcher.add_pattern(UnweightedPattern::from_graph(p1).unwrap());
        matcher.add_pattern(UnweightedPattern::from_graph(p2).unwrap());
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
        let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
        matcher.add_pattern(WeightedPattern::from_weighted_graph(p1, w1).unwrap());
        matcher.add_pattern(WeightedPattern::from_weighted_graph(p2, w2).unwrap());
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
        assert_eq!(matcher.find_weighted_matches(&g, &w).len(), 2);
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = UnweightedPattern::from_graph(pattern).unwrap();
            let mut matcher = TrieMatcher::new(TrieConstruction::Balanced);
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
        #[test]
        fn many_graphs_proptest_balanced(
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
                .map(|p| UnweightedPattern::from_graph(p).unwrap())
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
            let matcher = TrieMatcher::from_patterns(patterns.clone());
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
        #[test]
        fn many_graphs_proptest_balanced_optimised(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..100),
            g in gen_portgraph(30, 4, 60)
        ) {
            // for entry in glob("pattern_*.json").expect("glob pattern failed") {
            //     match entry {
            //         Ok(path) => fs::remove_file(path).expect("Removing file failed"),
            //         Err(_) => {},
            //     }
            // }
            // for (i, p) in patterns.iter().enumerate() {
            //     fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(p).unwrap()).unwrap();
            // }
            // fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
            let patterns = patterns
                .into_iter()
                .map(|p| UnweightedPattern::from_graph(p).unwrap())
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
            // fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
            let mut matcher = TrieMatcher::from_patterns(patterns.clone());
            matcher.optimise(10, 5);
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
        #[test]
        fn many_graphs_proptest_balanced_optimised_small(
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
                .map(|p| UnweightedPattern::from_graph(p).unwrap())
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
            // fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
            let mut matcher = TrieMatcher::from_patterns(patterns.clone());
            matcher.optimise(1, 10);
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
        #[test]
        fn many_graphs_proptest_non_det(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..10),
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
                .map(|p| UnweightedPattern::from_graph(p).unwrap())
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
            let mut matcher = TrieMatcher::new(TrieConstruction::NonDeterministic);
            for p in patterns.clone() {
                matcher.add_pattern(p);
            }
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
        #[test]
        fn many_graphs_proptest_det(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..10),
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
                .map(|p| UnweightedPattern::from_graph(p).unwrap())
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
            let mut matcher = TrieMatcher::new(TrieConstruction::Deterministic);
            for p in patterns.clone() {
                matcher.add_pattern(p);
            }
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
}
