//! An implementation of [`crate::Pattern`] based on patterns defined by a set
//! of constraints.

use itertools::Itertools;

use crate::{
    constraint::ConstraintSet,
    indexing::IndexKey,
    pattern::{Pattern, Satisfiable},
    Constraint, ConstraintClass, PartialPattern,
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

    /// Get the constraints in the pattern
    pub fn constraints(&self) -> &ConstraintSet<K, P> {
        &self.constraints
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
    P: ConditionalPredicate<K> + GetConstraintClass<K>,
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
        P::ConstraintClass::get_classes(&self)
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
    type Error = ();

    fn required_bindings(&self) -> Vec<Self::Key> {
        self.constraints
            .iter()
            .flat_map(|p| p.required_bindings().iter().copied())
            .unique()
            .collect()
    }

    fn try_into_partial_pattern(self) -> Result<Self::PartialPattern, ()> {
        Ok(self.into())
    }
}

impl<K, P> PartialPattern for PartialConstraintPattern<K, P>
where
    K: IndexKey,
    P: ConditionalPredicate<K> + GetConstraintClass<K>,
    P::ConstraintClass: Hash + Clone,
{
    type Constraint = Constraint<K, P>;

    type ConstraintClass = P::ConstraintClass;

    type Key = K;

    fn nominate(&self) -> impl Iterator<Item = Self::Constraint> + '_ {
        self.pattern_constraints.iter().cloned()
    }

    fn apply_transitions(
        &self,
        constraints: &[Self::Constraint],
        cls: &Self::ConstraintClass,
    ) -> Vec<Satisfiable<Self>> {
        if !self.all_classes().contains(cls) {
            // If no pattern constraints are in this class, we skip the
            // transition altogether
            return vec![];
        }
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
            tests::{TestConstraint, TestConstraintClass, TestKey, TestPattern, TestPredicate},
            ArityPredicate, GetConstraintClass,
        },
        constraint_class::ExpansionFactor,
        pattern::Satisfiable,
        Constraint, ConstraintClass, Pattern,
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
                .find(|c| c.get_classes() == self.try_get_classes(keys).unwrap());
            let Some(condition) = condition else {
                let self_constraint = self.clone().try_into_constraint(keys.to_vec()).unwrap();
                return Satisfiable::Yes(self_constraint);
            };
            assert_eq!(
                self.try_get_classes(keys).unwrap(),
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
            match self
                .try_get_classes(keys)
                .unwrap()
                .into_iter()
                .exactly_one()
                .unwrap()
            {
                TestConstraintClass::One(_, _) => {
                    // predicates are mutually exclusive
                    Satisfiable::No
                }
                TestConstraintClass::Two(_, _) => {
                    if condition.predicate() == &TestPredicate::AlwaysTrueTwo {
                        // Does not teach us anything, leave unchanged
                        Satisfiable::Yes(self.clone().try_into_constraint(keys.to_vec()).unwrap())
                    } else {
                        Satisfiable::Tautology
                    }
                }
                TestConstraintClass::Three => {
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
        type ConstraintClass = TestConstraintClass;
    }

    impl ConstraintClass<TestConstraint> for TestConstraintClass {
        fn get_classes(constraint: &TestConstraint) -> Vec<TestConstraintClass> {
            let keys = constraint.required_bindings();
            assert_eq!(constraint.arity(), keys.len());

            let args = keys.iter().cloned().collect_tuple();

            use TestPredicate::*;
            match constraint.predicate() {
                AreEqualOne | NotEqualOne => {
                    let (a, b) = args.unwrap();
                    vec![TestConstraintClass::One(a, b)]
                }
                AreEqualTwo | AlwaysTrueTwo => {
                    let (a, b) = args.unwrap();
                    vec![TestConstraintClass::Two(a, b)]
                }
                NeverTrueThree | AlwaysTrueThree => vec![TestConstraintClass::Three],
            }
        }

        fn expansion_factor<'c>(
            &self,
            _constraints: impl IntoIterator<Item = &'c TestConstraint>,
        ) -> ExpansionFactor
        where
            TestConstraint: 'c,
        {
            4
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
