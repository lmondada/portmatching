use crate::{
    constraint::Constraint,
    indexing::{DataKVMap, DataKey, IndexKey, IndexedData, Key},
    matcher::{PatternMatch, PortMatcher, SinglePatternMatcher},
    pattern::Pattern,
    IndexingScheme, Predicate,
};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
///
/// You probably do not want to use this matcher for anything other than as a
/// baseline in benchmarking.
pub struct NaiveManyMatcher<K, P, I> {
    matchers: Vec<SinglePatternMatcher<K, P, I>>,
}

impl<P, I: IndexingScheme + Default> NaiveManyMatcher<Key<I>, P, I> {
    /// Create a new naive matcher from patterns.
    ///
    /// Use [`IndexingScheme::default`] as the indexing scheme.
    pub fn try_from_patterns<'p, PT>(
        patterns: impl IntoIterator<Item = &'p PT>,
    ) -> Result<Self, PT::Error>
    where
        PT: Pattern<Constraint = Constraint<Key<I>, P>> + 'p,
    {
        let matchers = patterns
            .into_iter()
            .map(|p| SinglePatternMatcher::try_from_pattern(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { matchers })
    }

    /// Create a new naive matcher from patterns, using a custom indexing scheme.
    pub fn try_from_patterns_with_indexing<'p, PT>(
        patterns: impl IntoIterator<Item = &'p PT>,
        host_indexing: I,
    ) -> Result<Self, PT::Error>
    where
        I: Clone,
        PT: Pattern<Constraint = Constraint<Key<I>, P>> + 'p,
    {
        let matchers = patterns
            .into_iter()
            .map(|p| SinglePatternMatcher::try_from_pattern_with_indexing(p, host_indexing.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { matchers })
    }
}

impl<K: IndexKey, P, I> Default for NaiveManyMatcher<K, P, I> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<P, D> PortMatcher<D> for NaiveManyMatcher<DataKey<D>, P, <D as IndexedData>::IndexingScheme>
where
    D: IndexedData,
    P: Predicate<D>,
{
    type Match = DataKVMap<D>;

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

impl<K: IndexKey, P, I> FromIterator<SinglePatternMatcher<K, P, I>> for NaiveManyMatcher<K, P, I> {
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<K, P, I>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}
