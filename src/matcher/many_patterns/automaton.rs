use std::fmt::{self, Debug};
use std::hash::Hash;

use itertools::Itertools;

use crate::{
    automaton::{AutomatonBuilder, ConstraintAutomaton},
    constraint::{Constraint, DetHeuristic},
    indexing::IndexKey,
    matcher::PatternMatch,
    mutex_tree::ToConstraintsTree,
    pattern::Pattern,
    HashMap, IndexMap, IndexingScheme, PatternID, PortMatcher, Predicate,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<PT, K, P, I> {
    automaton: ConstraintAutomaton<Constraint<K, P>, I>,
    patterns: HashMap<PatternID, PT>,
}

impl<PT, K, P, D, I> PortMatcher<D> for ManyMatcher<PT, K, P, I>
where
    P: Predicate<D>,
    PT: Pattern<Constraint = Constraint<K, P>>,
    K: IndexKey,
    I: IndexingScheme<D>,
    I::Map: IndexMap<Key = K, Value = P::Value>,
{
    type Match = I::Map;

    fn find_matches<'a>(
        &'a self,
        host: &'a D,
    ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
        self.automaton.run(host).map_into()
    }
}

impl<K, P, PT, I> ManyMatcher<PT, K, P, I>
where
    Constraint<K, P>: Eq + Clone + Hash,
    P: ToConstraintsTree<K>,
    PT: Pattern<Constraint = Constraint<K, P>>,
    I: Default,
{
    /// Create a new matcher from patterns.
    ///
    /// The patterns are converted to constraints. Uses the deterministic
    /// heuristic provided by the constraint type.
    pub fn from_patterns(patterns: Vec<PT>) -> Self
    where
        P: DetHeuristic<K>,
    {
        Self::from_patterns_with_det_heuristic(patterns, P::make_det)
    }

    /// Create a new matcher from a vector of patterns, using a custom deterministic
    /// heuristic.
    pub fn from_patterns_with_det_heuristic(
        patterns: Vec<PT>,
        make_det: impl for<'c> FnMut(&[&'c Constraint<K, P>]) -> bool,
    ) -> Self {
        let constraints = patterns.iter().map(|p| p.to_constraint_vec()).collect_vec();
        let builder = AutomatonBuilder::from_constraints(constraints).set_det_heuristic(make_det);
        let (automaton, pattern_to_id) = builder.finish();
        let patterns = pattern_to_id.into_iter().zip(patterns).collect();
        Self {
            automaton,
            patterns,
        }
    }
}

impl<PT, K, P, I> Debug for ManyMatcher<PT, K, P, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<PT, K, P, I> ManyMatcher<PT, K, P, I> {
    /// Get a pattern by its ID.
    pub fn get_pattern(&self, id: PatternID) -> Option<&PT> {
        self.patterns.get(&id)
    }

    /// Get the number of states in the automaton.
    pub fn n_states(&self) -> usize {
        self.automaton.n_states()
    }

    /// Get the number of patterns in the matcher.
    pub fn n_patterns(&self) -> usize {
        self.patterns.len()
    }
}

impl<PT, K: Debug, P: Debug, I> ManyMatcher<PT, K, P, I> {
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}
