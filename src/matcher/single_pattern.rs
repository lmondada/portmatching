//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::collections::VecDeque;

use crate::{
    constraint::Constraint,
    indexing::{DataBindMap, DataKey, IndexKey, IndexedData, Key},
    pattern::Pattern,
    BindMap, HashSet, IndexingScheme, PatternID, Predicate,
};

use super::{PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<K, P, I> {
    /// The constraints forming the pattern
    constraints: Vec<Constraint<K, P>>,
    /// The indexing scheme for the host that we match on
    host_indexing: I,
    /// The bindings that must be present in the matches
    requested_bindings: HashSet<K>,
}

impl<P, D> PortMatcher<D>
    for SinglePatternMatcher<DataKey<D>, P, <D as IndexedData>::IndexingScheme>
where
    D: IndexedData,
    P: Predicate<D>,
{
    type Match = DataBindMap<D>;

    fn find_matches<'a>(
        &'a self,
        host: &'a D,
    ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
        self.get_all_bindings(host)
            .into_iter()
            .map(|bindings| PatternMatch::new(PatternID::default(), bindings))
    }
}

impl<P, I: IndexingScheme> SinglePatternMatcher<Key<I>, P, I> {
    /// Create a matcher from a vector of constraints.
    ///
    /// The host indexing scheme is the type's default.
    pub fn try_from_pattern<PT: Pattern<Key = Key<I>, Predicate = P>>(
        pattern: &PT,
    ) -> Result<Self, PT::Error>
    where
        I: Default,
    {
        Self::try_from_pattern_with_indexing(pattern, I::default())
    }

    /// Create a matcher from a vector of constraints with specified host indexing scheme.
    pub fn try_from_pattern_with_indexing<PT: Pattern<Key = Key<I>, Predicate = P>>(
        pattern: &PT,
        indexing: I,
    ) -> Result<Self, PT::Error> {
        let constraints = pattern.try_to_constraint_vec()?;
        let requested_bindings = indexing
            .all_missing_bindings(
                constraints
                    .iter()
                    .flat_map(|c| c.required_bindings().iter())
                    .copied(),
                [],
            )
            .into_iter()
            .collect();
        Ok(Self {
            constraints,
            host_indexing: indexing,
            requested_bindings,
        })
    }
}

impl<K: IndexKey, P, I: IndexingScheme> SinglePatternMatcher<K, P, I> {
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    pub fn match_exists<D>(&self, host: &D) -> bool
    where
        P: Predicate<D>,
        DataBindMap<D>: BindMap<Key = K>,
        D: IndexedData<IndexingScheme = I>,
    {
        !self.get_all_bindings(host).is_empty()
    }

    /// Get the valid scope assignments as a map from variable names to host nodes.
    fn get_all_bindings<D>(&self, host: &D) -> Vec<I::BindMap>
    where
        P: Predicate<D>,
        DataBindMap<D>: BindMap<Key = K>,
        D: IndexedData<IndexingScheme = I>,
    {
        let mut candidates = VecDeque::new();
        candidates.push_back((self.constraints.as_slice(), I::BindMap::default()));
        let mut final_bindings = Vec::new();
        while let Some((constraints, mut bindings)) = candidates.pop_front() {
            let [constraint, remaining @ ..] = constraints else {
                bindings.retain_keys(&self.requested_bindings);
                if self
                    .requested_bindings
                    .iter()
                    .all(|k| bindings.get(k).is_some())
                {
                    // We have a complete match
                    final_bindings.push(bindings);
                }
                continue;
            };

            let mut all_bindings = vec![bindings];
            let keys = self
                .host_indexing
                .all_missing_bindings(constraint.required_bindings().iter().copied(), []);
            all_bindings = all_bindings
                .into_iter()
                .flat_map(|bindings| host.bind_all(bindings, keys.iter().copied(), false))
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
        constraint::tests::TestConstraint,
        indexing::tests::{TestData, TestIndexingScheme},
        pattern::tests::TestPattern,
        predicate::tests::TestPredicate,
        HashMap,
    };

    use super::*;

    type TestMatcher = SinglePatternMatcher<usize, TestPredicate, TestIndexingScheme>;

    #[test]
    fn test_single_pattern_matcher() {
        let eq_2 = TestConstraint::new(vec![2, 2]);
        let pattern: TestPattern<usize, TestPredicate> = vec![eq_2].into();
        let matcher = TestMatcher::try_from_pattern(&pattern).unwrap();

        // Matching against itself works
        let matches = matcher.find_matches(&TestData).collect_vec();
        assert_eq!(matches.len(), 1);
        let PatternMatch {
            pattern,
            match_data,
        } = matches.first().unwrap();
        assert_eq!(match_data, &HashMap::from_iter((0..3).map(|i| (i, i))));
        assert_eq!(*pattern, PatternID::default());
    }
}
