//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
};

use itertools::Itertools;

use crate::{
    indexing::{bindings_hash, Binding, IndexKey, IndexedData},
    pattern::{PartialPattern, Pattern, Satisfiable},
    BindMap, Constraint, ConstraintEvaluator, EvaluateConstraints, HashSet, IndexingScheme,
    PatternID, Tag,
};

use super::{PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
#[derive(Debug, Clone)]
pub struct SinglePatternMatcher<K, B> {
    /// The constraints forming the pattern
    constraint_evaluators: Vec<B>,
    /// For each constraint evaluator, the keys that must be bound
    scopes: Vec<Vec<K>>,
    /// The bindings that must be present in the final matches
    required_bindings: BTreeSet<K>,
}

impl<B, D, K> PortMatcher<D> for SinglePatternMatcher<K, B>
where
    D: IndexedData<K>,
    B: EvaluateConstraints<D, D::Value, Key = K>,
    K: IndexKey,
{
    type Match = D::BindMap;

    fn find_matches<'a>(
        &'a self,
        host: &'a D,
    ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
        self.get_all_bindings(host)
            .into_iter()
            .map(|bindings| PatternMatch::new(PatternID::default(), bindings))
    }
}

impl<B: ConstraintEvaluator> SinglePatternMatcher<B::Key, B> {
    /// Create a matcher from a vector of constraints.
    ///
    /// The host indexing scheme is the type's default.
    pub fn try_from_pattern<I, PT>(pattern: PT) -> Result<Self, PT::Error>
    where
        PT: Pattern<Evaluator = B, Key = B::Key>,
        I: IndexingScheme<Key = B::Key> + Default,
    {
        Self::try_from_pattern_with_indexing(pattern, &I::default())
    }

    /// Create a matcher from a vector of constraints with specified host indexing scheme.
    pub fn try_from_pattern_with_indexing<PT>(
        pattern: PT,
        indexing: &impl IndexingScheme<Key = B::Key>,
    ) -> Result<Self, PT::Error>
    where
        PT: Pattern<Evaluator = B, Key = B::Key>,
    {
        // The set of keys that must be bound
        let required_bindings = BTreeSet::from_iter(pattern.required_bindings());

        // Break pattern into predicates
        let constraints = decompose_constraints(pattern.try_into_partial_pattern()?);

        // Turn predicates into constraint evaluator (with only one branch -- they
        // are just predicate evaluators in this case)
        let constraint_evaluators = constraints
            .into_iter()
            .map(|c| {
                let first_tag = c
                    .get_tags()
                    .into_iter()
                    .next()
                    .expect("must have at least one tag");
                first_tag.compile_evaluator([&c])
            })
            .collect_vec();

        // Compute each step of the way which bindings are required
        let mut scopes = Vec::<Vec<B::Key>>::with_capacity(constraint_evaluators.len());
        let mut known_bindings = HashSet::default();

        for br in &constraint_evaluators {
            let reqs = br.required_bindings().iter().copied();
            let new_keys = indexing.all_missing_bindings(reqs, known_bindings.iter().copied());
            known_bindings.extend(new_keys.iter().copied());
            scopes.push(new_keys);
        }

        // Add one more scope at the end for the final match
        let new_keys =
            indexing.all_missing_bindings(required_bindings.iter().copied(), known_bindings);
        scopes.push(new_keys);

        Ok(Self {
            constraint_evaluators,
            required_bindings,
            scopes,
        })
    }
}

fn decompose_constraints<P>(mut pattern: P) -> Vec<Constraint<P::Key, P::Predicate>>
where
    P: PartialPattern,
{
    match pattern.is_satisfiable() {
        Satisfiable::Yes(()) => {}
        Satisfiable::No => panic!("Pattern is not satisfiable"),
        Satisfiable::Tautology => return Vec::new(),
    }

    let mut all_constraints = vec![];
    let mut known_constraints = BTreeSet::new();
    loop {
        let Some(constraint) = find_best_constraint(&pattern) else {
            unimplemented!("SinglePatternMatcher currently only supports patterns that nominate a single constraint per tag");
        };
        let tag = constraint.get_tags().into_iter().next().unwrap();
        let new_patterns = pattern.apply_transitions(&[constraint.clone()], &tag);

        // Only support patterns with a single nominated constraint per tag
        let new_pattern = new_patterns
            .into_iter()
            .exactly_one()
            .ok()
            .expect("must match size of transitions");

        known_constraints.insert(constraint.clone());
        all_constraints.push(constraint);

        match new_pattern {
            Satisfiable::Yes(new_pattern) => pattern = new_pattern,
            Satisfiable::No => {
                panic!("Could not decompose pattern into constraints")
            }
            Satisfiable::Tautology => break,
        }
    }

    all_constraints
}

fn find_best_constraint<P: PartialPattern>(
    pattern: &P,
) -> Option<Constraint<P::Key, P::Predicate>> {
    let mut tag_to_constraints: BTreeMap<_, BTreeSet<_>> = BTreeMap::new();
    for c in pattern.nominate() {
        for cls in c.get_tags() {
            tag_to_constraints.entry(cls).or_default().insert(c.clone());
        }
    }
    tag_to_constraints.retain(|_, constraints| constraints.len() == 1);
    let (_, best_constraints) = tag_to_constraints
        .into_iter()
        .min_by_key(|(tag, constraints)| tag.expansion_factor(constraints))?;
    best_constraints.into_iter().exactly_one().ok()
}

impl<K: IndexKey, B: ConstraintEvaluator> SinglePatternMatcher<K, B> {
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    pub fn match_exists<D>(&self, host: &D) -> bool
    where
        B: EvaluateConstraints<D, D::Value, Key = K>,
        D: IndexedData<K>,
    {
        !self.get_all_bindings(host).is_empty()
    }

    /// Get the valid scope assignments as a map from variable names to host nodes.
    fn get_all_bindings<D>(&self, host: &D) -> Vec<D::BindMap>
    where
        B: EvaluateConstraints<D, D::Value, Key = K>,
        D: IndexedData<K>,
    {
        let mut all_bindings = vec![D::BindMap::default()];

        for (br, scope) in self.constraint_evaluators.iter().zip(&self.scopes) {
            // Add new required keys
            all_bindings = all_bindings
                .into_iter()
                .flat_map(|bindings| host.bind_all(bindings, scope.iter().copied()))
                .collect();

            // Keep the bindings that satisfy the predicate
            all_bindings.retain(|bindings| {
                let reqs = br.required_bindings();
                let bindings = reqs
                    .iter()
                    .map(|k| match bindings.get_binding(k) {
                        Binding::Bound(v) => Some(v.borrow().clone()),
                        Binding::Failed => None,
                        Binding::Unbound => panic!("tried to use unbound key {k:?}"),
                    })
                    .collect_vec();
                !br.eval(&bindings, host).is_empty()
            });
        }

        // Finally, process the last scope and create the matches
        let last_scope = BTreeSet::from_iter(self.scopes.last().unwrap().iter().copied());
        let mut final_bindings = all_bindings
            .into_iter()
            .flat_map(|bindings| host.bind_all(bindings, last_scope.iter().copied()))
            .collect_vec();

        for bindings in final_bindings.iter_mut() {
            bindings.retain_keys(&self.required_bindings);
        }
        final_bindings.retain(|bindings| {
            self.required_bindings
                .iter()
                .all(|k| bindings.get_binding(k).is_bound())
        });

        // Remove duplicates
        let mut hashes = BTreeSet::default();
        final_bindings.retain(|bindings| {
            let hash = bindings_hash(bindings, self.required_bindings.iter().copied());
            hashes.insert(hash)
        });

        final_bindings
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{
        constraint::{
            evaluator::tests::TestConstraintEvaluator,
            tests::{TestConstraint, TestPattern, TestPredicate},
        },
        indexing::tests::{TestData, TestStrIndexingScheme},
    };

    use super::*;

    type TestMatcher = SinglePatternMatcher<&'static str, TestConstraintEvaluator>;

    #[test]
    fn test_single_pattern_matcher() {
        let c1 = TestConstraint::try_binary_from_triple("key1", TestPredicate::AreEqualTwo, "key1")
            .unwrap();
        let c2 = TestConstraint::try_new(TestPredicate::AlwaysTrueThree, vec!["key100"]).unwrap();
        let c3 = TestConstraint::try_binary_from_triple("key1", TestPredicate::NotEqualOne, "key2")
            .unwrap();
        let pattern = TestPattern::from_constraints(vec![c1, c2, c3]);
        let matcher = TestMatcher::try_from_pattern::<TestStrIndexingScheme, _>(pattern).unwrap();

        let matches = matcher.find_matches(&TestData).collect_vec();

        assert_eq!(matches.len(), 1);
        insta::assert_debug_snapshot!(matches.first().unwrap())
    }
}
