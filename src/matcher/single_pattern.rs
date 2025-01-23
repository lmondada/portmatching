//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::{borrow::Borrow, collections::BTreeSet};

use itertools::Itertools;

use crate::{
    branch_selector::{BranchSelector, CreateBranchSelector, EvaluateBranchSelector},
    indexing::{bindings_hash, Binding, IndexKey, IndexedData},
    pattern::{Pattern, PatternLogic, Satisfiable},
    BindMap, HashSet, IndexingScheme, PatternID,
};

use super::{PatternMatch, PortMatcher};

/// A simple matcher for a single pattern.
#[derive(Debug, Clone)]
pub struct SinglePatternMatcher<K, B> {
    /// The constraints forming the pattern
    branch_selectors: Vec<B>,
    /// For each branch selector, the keys that must be bound
    scopes: Vec<Vec<K>>,
    /// The bindings that must be present in the final matches
    required_bindings: BTreeSet<K>,
}

impl<B, D, K> PortMatcher<D> for SinglePatternMatcher<K, B>
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
        self.get_all_bindings(host)
            .into_iter()
            .map(|bindings| PatternMatch::new(PatternID::default(), bindings))
    }
}

impl<K, B> SinglePatternMatcher<K, B> {
    /// Create a matcher from a vector of constraints.
    ///
    /// The host indexing scheme is the type's default.
    pub fn from_pattern<I, PT>(pattern: PT) -> Self
    where
        PT: Pattern<Key = K>,
        B: CreateBranchSelector<PT::Constraint, Key = K>,
        K: IndexKey,
        I: IndexingScheme<Key = K> + Default,
    {
        Self::from_pattern_with_indexing(pattern, &I::default())
    }

    /// Create a matcher from a vector of constraints with specified host indexing scheme.
    pub fn from_pattern_with_indexing<PT>(
        pattern: PT,
        indexing: &impl IndexingScheme<Key = K>,
    ) -> Self
    where
        PT: Pattern<Key = K>,
        B: CreateBranchSelector<PT::Constraint, Key = K>,
        K: IndexKey,
    {
        // The set of keys that must be bound
        let required_bindings = BTreeSet::from_iter(pattern.required_bindings());

        // Break pattern into predicates
        let constraints = decompose_constraints(pattern.into_logic());

        // Turn predicates into branch selectors (with only one branch -- they
        // are just predicate evaluators in this case)
        let branch_selectors = constraints
            .into_iter()
            .map(|p| B::create_branch_selector(vec![p]))
            .collect_vec();

        // Compute each step of the way which bindings are required
        let mut scopes = Vec::<Vec<K>>::with_capacity(branch_selectors.len());
        let mut known_bindings = HashSet::default();

        for br in &branch_selectors {
            let reqs = br.required_bindings().iter().copied();
            let new_keys = indexing.all_missing_bindings(reqs, known_bindings.iter().copied());
            known_bindings.extend(new_keys.iter().copied());
            scopes.push(new_keys);
        }

        // Add one more scope at the end for the final match
        let new_keys =
            indexing.all_missing_bindings(required_bindings.iter().copied(), known_bindings);
        scopes.push(new_keys);

        Self {
            branch_selectors,
            required_bindings,
            scopes,
        }
    }
}

fn decompose_constraints<P>(mut pattern: P) -> Vec<P::Constraint>
where
    P: PatternLogic,
{
    fn approx_isize(f: f64) -> isize {
        (f * 10000.) as isize
    }

    match pattern.is_satisfiable() {
        Satisfiable::Yes(()) => {}
        Satisfiable::No => panic!("Pattern is not satisfiable"),
        Satisfiable::Tautology => return Vec::new(),
    }

    let mut all_constraints = vec![];
    let mut known_constraints = BTreeSet::new();
    loop {
        let Some((cls, _)) = pattern
            .rank_classes(&[]) // TODO: keep track of known_bindings
            .max_by_key(|(_, rank)| approx_isize(*rank))
        else {
            return Vec::new();
        };

        let constraints = pattern.nominate(&cls).into_iter().collect_vec();
        let new_patterns = pattern.apply_transitions(&constraints);

        // Only support patterns with a single nominated constraint per class
        let Ok(constraint) = constraints.into_iter().exactly_one() else {
            unimplemented!("SinglePatternMatcher currently only supports patterns that nominate a single constraint per class");
        };
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

impl<K: IndexKey, B: BranchSelector> SinglePatternMatcher<K, B> {
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    pub fn match_exists<D>(&self, host: &D) -> bool
    where
        B: EvaluateBranchSelector<D, D::Value, Key = K>,
        D: IndexedData<K>,
    {
        !self.get_all_bindings(host).is_empty()
    }

    /// Get the valid scope assignments as a map from variable names to host nodes.
    fn get_all_bindings<D>(&self, host: &D) -> Vec<D::BindMap>
    where
        B: EvaluateBranchSelector<D, D::Value, Key = K>,
        D: IndexedData<K>,
    {
        let mut all_bindings = vec![D::BindMap::default()];

        for (br, scope) in self.branch_selectors.iter().zip(&self.scopes) {
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
        branch_selector::tests::TestBranchSelector,
        constraint::tests::{TestConstraint, TestPattern, TestPredicate},
        indexing::tests::{TestData, TestStrIndexingScheme},
    };

    use super::*;

    type TestMatcher = SinglePatternMatcher<&'static str, TestBranchSelector>;

    #[test]
    fn test_single_pattern_matcher() {
        let c1 = TestConstraint::try_binary_from_triple("key1", TestPredicate::AreEqualTwo, "key1")
            .unwrap();
        let c2 = TestConstraint::try_new(TestPredicate::AlwaysTrueThree, vec!["key100"]).unwrap();
        let c3 = TestConstraint::try_binary_from_triple("key1", TestPredicate::NotEqualOne, "key2")
            .unwrap();
        let pattern = TestPattern::from_constraints(vec![c1, c2, c3]);
        let matcher = TestMatcher::from_pattern::<TestStrIndexingScheme, _>(pattern);

        let matches = matcher.find_matches(&TestData).collect_vec();

        assert_eq!(matches.len(), 1);
        insta::assert_debug_snapshot!(matches.first().unwrap())
    }
}
