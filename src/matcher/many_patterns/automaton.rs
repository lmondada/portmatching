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
pub struct ManyMatcher<PT, K: IndexKey, P, I> {
    automaton: ConstraintAutomaton<K, P, I>,
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

/// What to do if a pattern fails to convert to a constraint.
#[derive(Debug, Clone, Copy, Default)]
pub enum PatternFallback {
    /// Skip the pattern.
    Skip,
    /// Fail if the pattern fails to convert to a constraint.
    #[default]
    Fail,
}

impl<K: IndexKey, P, PT, I> ManyMatcher<PT, K, P, I>
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
    pub fn try_from_patterns(
        patterns: Vec<PT>,
        fallback: PatternFallback,
    ) -> Result<Self, PT::Error>
    where
        P: DetHeuristic<K>,
    {
        Self::try_from_patterns_with_det_heuristic(patterns, P::make_det, fallback)
    }

    /// Create a new matcher from a vector of patterns, using a custom deterministic
    /// heuristic.
    pub fn try_from_patterns_with_det_heuristic(
        patterns: Vec<PT>,
        make_det: impl for<'c> FnMut(&[&'c Constraint<K, P>]) -> bool,
        fallback: PatternFallback,
    ) -> Result<Self, PT::Error> {
        let mut builder = AutomatonBuilder::new().set_det_heuristic(make_det);
        for (id, pattern) in patterns.iter().enumerate() {
            let constraints = match fallback {
                PatternFallback::Skip => {
                    let Ok(constraints) = pattern.try_to_constraint_vec() else {
                        continue;
                    };
                    constraints
                }
                PatternFallback::Fail => pattern.try_to_constraint_vec()?,
            };
            builder.add_pattern(constraints, id);
        }
        let (automaton, pattern_ids) = builder.finish();
        let patterns = patterns
            .into_iter()
            .enumerate()
            .map(|(i, p)| (PatternID(i), p))
            .filter(|(i, _)| pattern_ids.contains(&i))
            .collect();
        Ok(Self {
            automaton,
            patterns,
        })
    }
}

impl<PT, K: IndexKey, P, I> Debug for ManyMatcher<PT, K, P, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<PT, K: IndexKey, P, I> ManyMatcher<PT, K, P, I> {
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

impl<PT, K: IndexKey, P: Debug, I> ManyMatcher<PT, K, P, I> {
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}
