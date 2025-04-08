use crate::{
    branch_selector::{CreateBranchSelector, EvaluateBranchSelector},
    indexing::{IndexKey, IndexedData},
    matcher::{PatternMatch, PortMatcher, SinglePatternMatcher},
    pattern::{Pattern, PatternConstraint},
    IndexingScheme,
};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
///
/// You probably do not want to use this matcher for anything other than as a
/// baseline in benchmarking.
#[derive(Debug, Clone)]
pub struct NaiveManyMatcher<K, B> {
    matchers: Vec<SinglePatternMatcher<K, B>>,
}

impl<K, B> NaiveManyMatcher<K, B> {
    /// Create a new naive matcher from patterns.
    ///
    /// Use [`IndexingScheme::default`] as the indexing scheme.
    pub fn try_from_patterns<I, PT>(
        patterns: impl IntoIterator<Item = PT>,
    ) -> Result<Self, PT::Error>
    where
        PT: Pattern<Key = K>,
        B: CreateBranchSelector<PatternConstraint<PT>, Key = K>,
        K: IndexKey,
        I: IndexingScheme<Key = K> + Default,
    {
        let matchers = patterns
            .into_iter()
            .map(|p| SinglePatternMatcher::try_from_pattern::<I, PT>(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { matchers })
    }

    /// Create a new naive matcher from patterns, using a custom indexing scheme.
    pub fn try_from_patterns_with_indexing<PT>(
        patterns: impl IntoIterator<Item = PT>,
        host_indexing: &impl IndexingScheme<Key = K>,
    ) -> Result<Self, PT::Error>
    where
        PT: Pattern<Key = K>,
        B: CreateBranchSelector<PatternConstraint<PT>, Key = K>,
        K: IndexKey,
    {
        let matchers = patterns
            .into_iter()
            .map(|p| SinglePatternMatcher::try_from_pattern_with_indexing(p, host_indexing))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { matchers })
    }
}

impl<K, B> Default for NaiveManyMatcher<K, B> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<B, D, K> PortMatcher<D> for NaiveManyMatcher<K, B>
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
        self.matchers.iter().enumerate().flat_map(|(i, m)| {
            m.find_matches(host)
                .map(move |PatternMatch { match_data, .. }| PatternMatch {
                    pattern: i.into(),
                    match_data,
                })
        })
    }
}

impl<K, B> FromIterator<SinglePatternMatcher<K, B>> for NaiveManyMatcher<K, B> {
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<K, B>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}
