use std::fmt::{self, Debug};

use itertools::Itertools;

use crate::automaton::BuildConfig;
use crate::branch_selector::{CreateBranchSelector, DisplayBranchSelector, EvaluateBranchSelector};
use crate::indexing::IndexedData;
use crate::pattern::{Pattern, PatternConstraint};
use crate::IndexingScheme;
use crate::{
    automaton::{AutomatonBuilder, ConstraintAutomaton},
    indexing::IndexKey,
    matcher::PatternMatch,
    HashMap, PatternID, PortMatcher,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<PT, K: IndexKey, B> {
    automaton: ConstraintAutomaton<K, B>,
    patterns: HashMap<PatternID, PT>,
}

impl<PT, D, B, K> PortMatcher<D> for ManyMatcher<PT, K, B>
where
    K: IndexKey,
    D: IndexedData<K>,
    B: EvaluateBranchSelector<D, D::Value, Key = K>,
{
    type Match = D::BindMap;

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

impl<PT: Pattern + Clone, B> ManyMatcher<PT, PT::Key, B> {
    /// Create a new matcher from patterns.
    ///
    /// The patterns are converted to constraints. Uses the deterministic
    /// heuristic provided by the constraint type.
    pub fn try_from_patterns<I>(
        patterns: Vec<PT>,
        fallback: PatternFallback,
    ) -> Result<Self, PT::Error>
    where
        I: IndexingScheme<Key = PT::Key> + Default,
        B: CreateBranchSelector<PatternConstraint<PT>, Key = PT::Key>,
    {
        Self::try_from_patterns_with_config(patterns, BuildConfig::<I>::default(), fallback)
    }

    /// Create a new matcher from a vector of patterns, using a custom deterministic
    /// heuristic.
    pub fn try_from_patterns_with_config(
        patterns: Vec<PT>,
        config: BuildConfig<impl IndexingScheme<Key = PT::Key>>,
        fallback: PatternFallback,
    ) -> Result<Self, PT::Error>
    where
        B: CreateBranchSelector<PatternConstraint<PT>, Key = PT::Key>,
    {
        let builder = AutomatonBuilder::try_from_patterns(patterns.iter().cloned(), fallback)?;
        let (automaton, ids) = builder.build(config);

        let patterns = patterns
            .into_iter()
            .zip(ids)
            .map(|(p, id)| (id, p))
            .collect();

        Ok(Self {
            automaton,
            patterns,
        })
    }
}

impl<PT, K: IndexKey, B> Debug for ManyMatcher<PT, K, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<PT, K: IndexKey, B> ManyMatcher<PT, K, B> {
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

impl<PT, K: IndexKey, B: DisplayBranchSelector> ManyMatcher<PT, K, B> {
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}
