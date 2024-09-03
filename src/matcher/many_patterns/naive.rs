use crate::{
    constraint::Constraint,
    indexing,
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
pub struct NaiveManyMatcher<C, I> {
    matchers: Vec<SinglePatternMatcher<C, I>>,
}

impl<C, I> NaiveManyMatcher<C, I> {
    /// Create a new naive matcher from patterns.
    ///
    /// Use [`IndexingScheme::default`] as the indexing scheme.
    pub fn try_from_patterns<'p, PT>(
        patterns: impl IntoIterator<Item = &'p PT>,
    ) -> Result<Self, PT::Error>
    where
        PT: Pattern<Constraint = C> + 'p,
        I: Default,
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
        PT: Pattern<Constraint = C> + 'p,
    {
        let matchers = patterns
            .into_iter()
            .map(|p| SinglePatternMatcher::try_from_pattern_with_indexing(p, host_indexing.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { matchers })
    }
}

impl<C, I> Default for NaiveManyMatcher<C, I> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<P, D, I> PortMatcher<D> for NaiveManyMatcher<Constraint<indexing::Key<I, D>, P>, I>
where
    P: Predicate<D, Value = indexing::Value<I, D>>,
    I: IndexingScheme<D>,
{
    type Match = I::Map;

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

impl<C, I> FromIterator<SinglePatternMatcher<C, I>> for NaiveManyMatcher<C, I> {
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<C, I>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}
