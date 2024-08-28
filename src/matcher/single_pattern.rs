//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::collections::VecDeque;

use crate::{
    constraint::Constraint,
    indexing::{self, IndexKey},
    pattern::Pattern,
    IndexMap, IndexingScheme, PatternID, Predicate,
};

use super::{PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<C, I> {
    /// The constraints forming the pattern
    constraints: Vec<C>,
    /// The indexing scheme for the host that we match on
    host_indexing: I,
}

impl<P, D, I> PortMatcher<D> for SinglePatternMatcher<Constraint<indexing::Key<I, D>, P>, I>
where
    P: Predicate<D, Value = indexing::Value<I, D>>,
    I: IndexingScheme<D>,
{
    type Match = I::Map;

    fn find_matches<'a>(
        &'a self,
        host: &'a D,
    ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
        self.get_all_bindings(host)
            .into_iter()
            .map(|bindings| PatternMatch::new(PatternID::default(), bindings))
    }
}

impl<C, I> SinglePatternMatcher<C, I> {
    /// Create a matcher from a vector of constraints.
    ///
    /// The host indexing scheme is the type's default.
    pub fn try_from_pattern<PT: Pattern<Constraint = C>>(pattern: &PT) -> Result<Self, PT::Error>
    where
        I: Default,
    {
        Self::try_from_pattern_with_indexing(pattern, I::default())
    }

    /// Create a matcher from a vector of constraints with specified host indexing scheme.
    pub fn try_from_pattern_with_indexing<PT: Pattern<Constraint = C>>(
        pattern: &PT,
        indexing: I,
    ) -> Result<Self, PT::Error> {
        let constraints = pattern.try_to_constraint_vec()?;
        Ok(Self {
            constraints,
            host_indexing: indexing,
        })
    }
}

impl<K, P, I> SinglePatternMatcher<Constraint<K, P>, I>
where
    K: IndexKey,
{
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    pub fn match_exists<D>(&self, host: &D) -> bool
    where
        P: Predicate<D, Value = indexing::Value<I, D>>,
        I: IndexingScheme<D>,
        I::Map: IndexMap<Key = K>,
    {
        !self.get_all_bindings(host).is_empty()
    }

    /// Get the valid scope assignments as a map from variable names to host nodes.
    fn get_all_bindings<D>(&self, host: &D) -> Vec<I::Map>
    where
        P: Predicate<D, Value = indexing::Value<I, D>>,
        I: IndexingScheme<D>,
        I::Map: IndexMap<Key = K>,
    {
        let mut candidates = VecDeque::new();
        candidates.push_back((self.constraints.as_slice(), I::Map::default()));
        let mut final_bindings = Vec::new();
        while let Some((constraints, bindings)) = candidates.pop_front() {
            let [constraint, remaining @ ..] = constraints else {
                final_bindings.push(bindings);
                continue;
            };

            let mut all_bindings = vec![bindings];
            let keys = constraint.required_bindings();
            all_bindings = all_bindings
                .into_iter()
                .flat_map(|bindings| {
                    self.host_indexing
                        .try_bind_all(bindings, keys.to_vec(), host)
                        .unwrap()
                })
                .collect();
            candidates.extend(
                all_bindings
                    .into_iter()
                    .filter(|bindings| constraint.is_satisfied(host, bindings).unwrap_or(false))
                    .map(|bindings| (remaining, bindings)),
            )
        }
        final_bindings
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{
        constraint::tests::TestConstraint, indexing::tests::TestIndexingScheme,
        pattern::tests::TestPattern, HashMap,
    };

    use super::*;

    type TestMatcher = SinglePatternMatcher<TestConstraint, TestIndexingScheme>;

    #[test]
    fn test_single_pattern_matcher() {
        let eq_2 = TestConstraint::new(vec![2, 2]);
        let pattern: TestPattern<TestConstraint> = vec![eq_2].into();
        let matcher = TestMatcher::try_from_pattern(&pattern).unwrap();

        // Matching against itself works
        let matches = matcher.find_matches(&()).collect_vec();
        assert_eq!(matches.len(), 1);
        let PatternMatch {
            pattern,
            match_data,
        } = matches.first().unwrap();
        assert_eq!(match_data, &HashMap::from_iter((0..3).map(|i| (i, i))));
        assert_eq!(*pattern, PatternID::default());
    }
}
