use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use portgraph::{NodeIndex, PortGraph, PortIndex};

use crate::pattern::{Edge, Pattern};

use super::Matcher;

mod naive;
pub use naive::{NaiveGraphTrie, NaiveManyPatternMatcher};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternMatch {
    id: PatternID,
    root: NodeIndex,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternID(usize);

impl fmt::Debug for PatternID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub trait ReadGraphTrie {
    type StateID;
    type MatchObject: Clone;

    fn init(&self, root: NodeIndex) -> (Self::StateID, Self::MatchObject);

    fn next_states(
        &self,
        state: &Self::StateID,
        graph: &PortGraph,
        current_match: &Self::MatchObject,
    ) -> Vec<(Self::StateID, Self::MatchObject)>;
}

pub trait WriteGraphTrie: ReadGraphTrie {
    fn create_next_states(
        &mut self,
        state: &Self::StateID,
        edge: (PortIndex, PortIndex),
        graph: &PortGraph,
        current_match: &Self::MatchObject,
    ) -> Vec<(Self::StateID, Self::MatchObject)>;

    fn create_next_roots(
        &mut self,
        state: &Self::StateID,
        current_match: &Self::MatchObject,
        is_dangling: bool,
    ) -> Vec<(Self::StateID, Self::MatchObject)>;
}

pub struct ManyPatternMatcher<T: ReadGraphTrie> {
    trie: T,
    patterns: Vec<Pattern>,
    matching_nodes: BTreeMap<T::StateID, Vec<PatternID>>,
}

impl<T: Default + ReadGraphTrie> Default for ManyPatternMatcher<T> {
    fn default() -> Self {
        Self {
            trie: T::default(),
            patterns: Vec::new(),
            matching_nodes: BTreeMap::new(),
        }
    }
}

impl<T: Default + ReadGraphTrie> ManyPatternMatcher<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: ReadGraphTrie> Matcher for ManyPatternMatcher<T>
where
    T::StateID: Ord,
{
    type Match = PatternMatch;

    fn find_anchored_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Self::Match> {
        let mut current_states = vec![self.trie.init(root)];
        let mut matches = BTreeSet::new();
        while !current_states.is_empty() {
            let mut new_states = Vec::new();
            for (state, current_match) in current_states {
                for &pattern_id in self.matching_nodes.get(&state).unwrap_or(&[].to_vec()) {
                    matches.insert(PatternMatch {
                        id: pattern_id,
                        root,
                    });
                }
                for (next_state, next_match) in self.trie.next_states(&state, graph, &current_match)
                {
                    new_states.push((next_state, next_match));
                }
            }
            current_states = new_states;
        }
        Vec::from_iter(matches)
    }
}

impl<T: WriteGraphTrie + Default> ManyPatternMatcher<T>
where
    T::StateID: Ord,
{
    pub fn from_patterns(patterns: Vec<Pattern>) -> Self {
        let mut obj = Self {
            trie: T::default(),
            patterns: Vec::with_capacity(patterns.len()),
            matching_nodes: BTreeMap::new(),
        };
        for pattern in patterns {
            obj.add_pattern(pattern);
        }
        obj
    }
}

impl<T: WriteGraphTrie> ManyPatternMatcher<T>
where
    T::StateID: Ord,
{
    pub fn add_pattern(&mut self, pattern: Pattern) -> PatternID {
        // The pattern number of this pattern
        let pattern_id = PatternID(self.patterns.len());
        self.patterns.push(pattern);
        let pattern = &self.patterns[pattern_id.0];
        let graph = &pattern.graph;

        // Stores the current positions in the graph trie, along with the
        // match that corresponds to that position
        let mut current_states = vec![self.trie.init(pattern.root)];

        // Decompose a pattern into "lines", which are paths in the pattern
        let all_lines = pattern.all_lines();

        // The only edge that can be "dangling" (i.e. have no second endvertex)
        // in a line is its last edge.
        // This is `None` at the beginning (no previous line), and then will
        // always be Some(true/false)
        let mut is_previous_line_dangling = None;
        for line in all_lines {
            // Move on from previous line
            // We do this at the beginning of the loop and not the end because
            // we do not want to move on after the last line
            if let Some(is_dangling) = is_previous_line_dangling {
                let mut new_states = Vec::new();
                for (state, current_match) in current_states {
                    new_states.append(&mut self.trie.create_next_roots(
                        &state,
                        &current_match,
                        is_dangling,
                    ));
                }
                current_states = new_states;
            }
            // Traverse the line
            is_previous_line_dangling = false.into();
            for edge in line {
                let Edge(out_port, Some(in_port)) = edge else {
                    is_previous_line_dangling = true.into();
                    break;
                };
                let mut new_states = Vec::new();
                for (state, current_match) in current_states {
                    new_states.append(&mut self.trie.create_next_states(
                        &state,
                        (out_port, in_port),
                        graph,
                        &current_match,
                    ));
                }
                current_states = new_states;
            }
        }
        for (state, _) in current_states {
            self.matching_nodes
                .entry(state)
                .or_insert_with(|| Vec::new())
                .push(pattern_id);
        }
        pattern_id
    }
}
