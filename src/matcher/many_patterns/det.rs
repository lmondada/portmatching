use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
};

use portgraph::{dot::dot_string_weighted, NodeIndex, PortGraph};

use crate::{
    constraint::{Constraint, Skeleton, UnweightedConstraint},
    graph_tries::{root_state, BaseGraphTrie, GraphTrie, StateID},
    pattern::Edge,
    ManyPatternMatcher, Matcher, Pattern, PatternID,
};

use super::PatternMatch;

/// A graph trie matcher using only deterministic states
///
/// In theory, this should perform best, but trie sizes grow exponentially,
/// so not recommended in practice.
pub struct DetTrieMatcher<T> {
    trie: T,
    match_states: BTreeMap<StateID, Vec<PatternID>>,
    patterns: Vec<Pattern>,
}

impl<T> Matcher for DetTrieMatcher<T>
where
    T: GraphTrie,
{
    type Match = PatternMatch;

    fn find_anchored_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Self::Match> {
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
                for next_state in self.trie.next_states(state, graph, root) {
                    new_states.push(next_state);
                }
            }
            current_states = new_states;
        }
        Vec::from_iter(matches)
    }
}

impl<C: Constraint + Clone> Default for DetTrieMatcher<BaseGraphTrie<C>> {
    fn default() -> Self {
        Self {
            trie: Default::default(),
            match_states: Default::default(),
            patterns: Default::default(),
        }
    }
}

impl ManyPatternMatcher for DetTrieMatcher<BaseGraphTrie<UnweightedConstraint>> {
    fn add_pattern(&mut self, pattern: Pattern) -> PatternID {
        // The pattern number of this pattern
        let pattern_id = PatternID(self.patterns.len());
        self.patterns.push(pattern);
        let pattern = &self.patterns[pattern_id.0];
        let graph = &pattern.graph;
        let skeleton = Skeleton::new(graph, pattern.root);

        // Stores the current positions in the graph trie, along with the
        // match that corresponds to that position
        let mut current_states: BTreeSet<_> = [root_state()].into();

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

        for Edge(out_port, in_port) in pattern.canonical_edge_ordering() {
            // All other edges are deterministic
            let constraint = if let Some(in_port) = in_port {
                UnweightedConstraint::Adjacency {
                    other_ports: skeleton.get_port_addr(in_port),
                }
            } else {
                UnweightedConstraint::Dangling
            };
            current_states = self.trie.add_graph_edge_det(
                &skeleton.get_port_addr(out_port),
                current_states,
                constraint,
            );
        }

        let current_states = self.trie.finalize(clone_state);

        // Record matching pattern in final states
        for state in current_states {
            self.match_states.entry(state).or_default().push(pattern_id);
        }

        pattern_id
    }
}

impl<C: Constraint + Display> DetTrieMatcher<BaseGraphTrie<C>>
where
    C::Address: Debug,
{
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

#[cfg(test)]
mod tests {
    use super::DetTrieMatcher;
    use crate::{
        matcher::{
            many_patterns::{ManyPatternMatcher, PatternID, PatternMatch},
            Matcher, SinglePatternMatcher,
        },
        pattern::Pattern,
        utils::test_utils::gen_portgraph_connected,
    };
    use portgraph::proptest::gen_portgraph;
    use proptest::prelude::*;

    use itertools::Itertools;

    proptest! {
        #[ignore = "a bit slow"]
        #[test]
        fn many_graphs_proptest_det_trie(
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
            let matcher = DetTrieMatcher::from_patterns(patterns.clone());
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
