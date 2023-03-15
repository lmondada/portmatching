use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use portgraph::{NodeIndex, PortGraph};

use crate::pattern::{Edge, Pattern};

use super::Matcher;

mod naive;
pub use naive::{NaiveGraphTrie, NaiveManyPatternMatcher};

/// A match instance returned by a ManyPatternMatcher instance
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternMatch {
    pub id: PatternID,
    pub root: NodeIndex,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternID(usize);

impl fmt::Debug for PatternID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// The Graph Trie trait
///
/// Any struct that implements this trait can be used as a graph trie for
/// pattern matching.
/// - `init` should return the root of the trie and an empty match object
/// - `next_states` should return the set of states that can be obtained
///    by a single transition from `state`
///
/// A graph trie is thus a non-deterministic automaton.
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

/// A Graph Trie that supports the addition of new patterns
///
/// Writing new patterns to the graph trie requires two functions
/// - `create_next_states` is the equivalent of `next_states` in write mode
/// - `create_next_roots` is basically a carriage return, to be called at the
///   end of the line.
pub trait WriteGraphTrie: ReadGraphTrie {
    fn create_next_states<F: FnMut(&Self::StateID, &Self::StateID)>(
        &mut self,
        state: &Self::StateID,
        edge: &Edge,
        graph: &PortGraph,
        current_match: &Self::MatchObject,
        clone_state: F,
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
    T::StateID: Ord + Clone,
{
    /// Construct a Graph Trie from a vector of patterns
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
    T::StateID: Ord + Clone,
{
    /// Add a pattern to the graph trie
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

        for line in all_lines {
            // The only edge that can be "dangling" (i.e. have no second endvertex)
            // in a line is its last edge.
            // We store this as it changes how we can proceed to the next line.
            let mut is_dangling = false;

            // A callback when a state is cloned in the trie
            // necessary to keep track of the match states
            let mut clone_state = |old_state: &T::StateID, new_state: &T::StateID| {
                self.matching_nodes.insert(
                    new_state.clone(),
                    self.matching_nodes
                        .get(old_state)
                        .cloned()
                        .unwrap_or_default(),
                );
            };

            // Traverse the line
            for edge in line {
                let mut new_states = Vec::new();
                for (state, current_match) in current_states {
                    new_states.append(&mut self.trie.create_next_states(
                        &state,
                        &edge,
                        graph,
                        &current_match,
                        &mut clone_state,
                    ));
                }
                current_states = new_states;
                if edge.1.is_none() {
                    is_dangling = true;
                }
            }

            // Move on to next line
            // It is important to "move on to next line" even if it is
            // the last line in the case of dangling edges, to make sure
            // we match these.
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

        // Record matching pattern in final states
        for (state, _) in current_states {
            self.matching_nodes
                .entry(state)
                .or_insert_with(Vec::new)
                .push(pattern_id);
        }
        pattern_id
    }
}
