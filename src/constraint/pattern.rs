//! An implementation of [`crate::Pattern`] based on patterns defined by a set
//! of constraints.

use itertools::Itertools;

use crate::{
    constraint::ConstraintSet,
    indexing::IndexKey,
    pattern::{ClassRank, Pattern, Satisfiable},
    Constraint, PartialPattern,
};
use std::{collections::BTreeSet, hash::Hash};

use super::{ConditionalPredicate, GetConstraintClass};

/// A pattern that is defined by a set of constraints.
///
/// The pattern is considered to match if and only if all its constraints are
/// satisfied.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConstraintPattern<K, P> {
    constraints: ConstraintSet<K, P>,
}

/// A partially satisfied [`ConstraintPattern`].
///
/// This keeps track of constraints that are known to be satisfied, along
/// with the remaining pattern constraints yet to be satisfied.
///
/// Provides an implementation of [`PartialPattern`]. See [`PartialPattern`] for
/// more information.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PartialConstraintPattern<K, P> {
    pattern_constraints: ConstraintSet<K, P>,
    known_constraints: BTreeSet<Constraint<K, P>>,
}

impl<K, P> ConstraintPattern<K, P>
where
    P: ConditionalPredicate<K>,
{
    /// Create a new [`ConstraintPattern`] from a set of constraints
    ///
    /// Every constraint must have a different class, otherwise this will panic.
    pub fn from_constraints_set(constraints: BTreeSet<Constraint<K, P>>) -> Self {
        Self { constraints }
    }

    /// Create a new [`ConstraintPattern`] from an iterator of constraints
    pub fn from_constraints(constraints: impl IntoIterator<Item = Constraint<K, P>>) -> Self
    where
        K: Ord,
    {
        Self::from_constraints_set(constraints.into_iter().collect())
    }
}

impl<K, P> Into<PartialConstraintPattern<K, P>> for ConstraintPattern<K, P> {
    fn into(self) -> PartialConstraintPattern<K, P> {
        PartialConstraintPattern {
            pattern_constraints: self.constraints,
            known_constraints: BTreeSet::new(),
        }
    }
}

impl<K, P> PartialConstraintPattern<K, P>
where
    P: GetConstraintClass<K> + ConditionalPredicate<K>,
{
    /// Get all unique constraint classes from the constraints
    pub fn all_classes(&self) -> BTreeSet<P::ConstraintClass> {
        self.pattern_constraints
            .iter()
            .flat_map(|p| p.get_classes())
            .collect()
    }

    /// Get constraints that match a specific constraint class
    ///
    /// # Arguments
    /// * `cls` - The constraint class to match
    pub fn get_by_class<'c>(
        &'c self,
        cls: &'c P::ConstraintClass,
    ) -> impl Iterator<Item = &'c Constraint<K, P>> {
        self.pattern_constraints
            .iter()
            .filter(|p| p.get_classes().contains(cls))
    }

    /// Condition pattern on a set of known constraints
    pub fn conditioned(
        &self,
        known_constraints: BTreeSet<Constraint<K, P>>,
        prev_constraints: &[Constraint<K, P>],
    ) -> Satisfiable<Self>
    where
        K: IndexKey,
    {
        let mut new_constraints = BTreeSet::new();

        for c in &self.pattern_constraints {
            match c.condition_on(&known_constraints, prev_constraints) {
                Satisfiable::Yes(new_c) => {
                    new_constraints.insert(new_c);
                }
                Satisfiable::Tautology => (),
                Satisfiable::No => return Satisfiable::No,
            }
        }

        if new_constraints.is_empty() {
            return Satisfiable::Tautology;
        }

        Satisfiable::Yes(Self {
            pattern_constraints: new_constraints,
            known_constraints,
        })
    }
}

impl<K, P> Constraint<K, P> {
    /// Get the constraint classes that this constraint belongs to
    pub fn get_classes(&self) -> Vec<P::ConstraintClass>
    where
        P: GetConstraintClass<K>,
    {
        self.predicate().get_classes(self.required_bindings())
    }

    /// Condition this constraint on a set of known constraints
    ///
    /// # Arguments
    /// * `known_constraints` - The set of known constraints
    /// * `prev_constraints` - The set of constraints that have been evaluated
    ///                        so far
    pub fn condition_on(
        &self,
        known_constraints: &BTreeSet<Constraint<K, P>>,
        prev_constraints: &[Constraint<K, P>],
    ) -> Satisfiable<Constraint<K, P>>
    where
        P: ConditionalPredicate<K>,
    {
        let keys = self.required_bindings();
        self.predicate()
            .condition_on(keys, known_constraints, prev_constraints)
    }
}

impl<K, P> Pattern for ConstraintPattern<K, P>
where
    K: IndexKey,
    P: GetConstraintClass<K> + ConditionalPredicate<K>,
    P::ConstraintClass: Hash + Clone,
{
    type Key = K;

    type PartialPattern = PartialConstraintPattern<K, P>;

    type Constraint = Constraint<K, P>;

    fn required_bindings(&self) -> Vec<Self::Key> {
        self.constraints
            .iter()
            .flat_map(|p| p.required_bindings().iter().copied())
            .unique()
            .collect()
    }

    fn into_partial_pattern(self) -> Self::PartialPattern {
        self.into()
    }
}

impl<K, P> PartialPattern for PartialConstraintPattern<K, P>
where
    K: IndexKey,
    P: ConditionalPredicate<K> + GetConstraintClass<K>,
    P::ConstraintClass: Hash + Clone,
{
    type Constraint = Constraint<K, P>;

    type BranchClass = P::ConstraintClass;

    type Key = K;

    fn rank_classes(
        &self,
        _: &[Self::Key],
    ) -> impl Iterator<Item = (Self::BranchClass, ClassRank)> {
        self.all_classes().into_iter().map(|cls| (cls, 0.5))
    }

    fn nominate(&self, cls: &Self::BranchClass) -> BTreeSet<Self::Constraint> {
        self.pattern_constraints
            .iter()
            .filter(|c| c.get_classes().iter().any(|c| c == cls))
            .cloned()
            .collect()
    }

    fn apply_transitions(&self, constraints: &[Self::Constraint]) -> Vec<Satisfiable<Self>> {
        let mut conditioned = Vec::with_capacity(constraints.len());

        for (i, c) in constraints.iter().enumerate() {
            let mut known_constraints = self.known_constraints.clone();
            known_constraints.insert(c.clone());
            let prev_constraints = &constraints[..i];
            conditioned.push(self.conditioned(known_constraints, prev_constraints));
        }

        conditioned
    }

    fn is_satisfiable(&self) -> Satisfiable {
        if self.pattern_constraints.is_empty() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use itertools::Itertools;

    use crate::{
        constraint::{
            tests::{TestBranchClass, TestConstraint, TestKey, TestPattern, TestPredicate},
            ArityPredicate, GetConstraintClass,
        },
        pattern::Satisfiable,
        Constraint, Pattern,
    };

    use super::ConditionalPredicate;

    impl ConditionalPredicate<TestKey> for TestPredicate {
        fn condition_on(
            &self,
            keys: &[TestKey],
            known_constraints: &BTreeSet<Constraint<TestKey, Self>>,
            _: &[Constraint<TestKey, Self>],
        ) -> Satisfiable<Constraint<TestKey, Self>> {
            let condition = known_constraints
                .iter()
                .find(|c| c.get_classes() == self.get_classes(keys));
            let Some(condition) = condition else {
                let self_constraint = self.clone().try_into_constraint(keys.to_vec()).unwrap();
                return Satisfiable::Yes(self_constraint);
            };
            assert_eq!(
                self.get_classes(keys),
                condition.get_classes(),
                "class mismatch in TestPredicate::condition_on"
            );
            assert_eq!(
                keys,
                condition.required_bindings(),
                "Cannot be in same class if keys are not the same"
            );

            if self == condition.predicate() {
                return Satisfiable::Tautology;
            }
            match self.get_classes(keys).into_iter().exactly_one().unwrap() {
                TestBranchClass::One(_, _) => {
                    // predicates are mutually exclusive
                    Satisfiable::No
                }
                TestBranchClass::Two(_, _) => {
                    if condition.predicate() == &TestPredicate::AlwaysTrueTwo {
                        // Does not teach us anything, leave unchanged
                        Satisfiable::Yes(self.clone().try_into_constraint(keys.to_vec()).unwrap())
                    } else {
                        Satisfiable::Tautology
                    }
                }
                TestBranchClass::Three => {
                    if condition.predicate() == &TestPredicate::AlwaysTrueThree {
                        // Does not teach us anything, leave unchanged
                        Satisfiable::Yes(self.clone().try_into_constraint(keys.to_vec()).unwrap())
                    } else {
                        Satisfiable::Tautology
                    }
                }
            }
        }
    }

    impl GetConstraintClass<TestKey> for TestPredicate {
        type ConstraintClass = TestBranchClass;

        fn get_classes(&self, keys: &[TestKey]) -> Vec<Self::ConstraintClass> {
            assert_eq!(self.arity(), keys.len());

            let args = keys.iter().cloned().collect_tuple();

            use TestPredicate::*;
            match self {
                AreEqualOne | NotEqualOne => {
                    let (a, b) = args.unwrap();
                    vec![TestBranchClass::One(a, b)]
                }
                AreEqualTwo | AlwaysTrueTwo => {
                    let (a, b) = args.unwrap();
                    vec![TestBranchClass::Two(a, b)]
                }
                NeverTrueThree | AlwaysTrueThree => vec![TestBranchClass::Three],
            }
        }
    }

    #[test]
    fn test_required_bindings() {
        let c = TestConstraint::try_binary_from_triple("key1", TestPredicate::AreEqualOne, "key1")
            .unwrap();
        let p = TestPattern::from_constraints(vec![c]);

        assert_eq!(p.required_bindings(), vec!["key1"]);
    }
}
