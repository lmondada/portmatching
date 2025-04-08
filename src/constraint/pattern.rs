//! An implementation of [`crate::Pattern`] based on patterns defined by a set
//! of constraints.

use itertools::Itertools;

use crate::{
    constraint::ConstraintSet,
    indexing::IndexKey,
    pattern::{PartialPatternTag, Pattern, Satisfiable},
    Constraint, PartialPattern,
};
use std::{collections::BTreeSet, hash::Hash};

use super::{ConditionalPredicate, ConstraintTag};

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
    /// Every constraint must have a different tag, otherwise this will panic.
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
    P: ConditionalPredicate<K> + ConstraintTag<K>,
{
    /// Get all unique constraint tags from the constraints
    pub fn all_tags(&self) -> BTreeSet<P::Tag> {
        self.pattern_constraints
            .iter()
            .flat_map(|p| p.get_tags())
            .collect()
    }

    /// Get constraints that match a specific constraint tag
    ///
    /// # Arguments
    /// * `tag` - The constraint tag to match
    pub fn get_by_tag<'c>(&'c self, tag: &'c P::Tag) -> impl Iterator<Item = &'c Constraint<K, P>> {
        self.pattern_constraints
            .iter()
            .filter(|p| p.get_tags().contains(tag))
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
    /// Get the constraint tags that this constraint belongs to
    pub fn get_tags(&self) -> Vec<P::Tag>
    where
        P: ConstraintTag<K>,
    {
        P::get_tags(&self.predicate(), &self.args)
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
    P: ConstraintTag<K> + ConditionalPredicate<K>,
    P::Tag: Hash + Clone,
{
    type Key = K;
    type PartialPattern = PartialConstraintPattern<K, P>;
    type Predicate = P;
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
    P: ConditionalPredicate<K> + ConstraintTag<K>,
    P::Tag: Hash + Clone,
{
    type Key = K;
    type Predicate = P;

    fn nominate(&self) -> impl Iterator<Item = Constraint<Self::Key, Self::Predicate>> + '_ {
        self.pattern_constraints.iter().cloned()
    }

    fn apply_transitions(
        &self,
        constraints: &[Constraint<Self::Key, Self::Predicate>],
        cls: &PartialPatternTag<Self>,
    ) -> Vec<Satisfiable<Self>> {
        if !self.all_tags().contains(cls) {
            // If no pattern constraints are in this tag, we skip the
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
            tests::{TestConstraint, TestConstraintTag, TestKey, TestPattern, TestPredicate},
            ArityPredicate, ConstraintTag, Tag,
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
                .find(|c| c.get_tags() == self.get_tags(keys));
            let Some(condition) = condition else {
                let self_constraint = self.clone().try_into_constraint(keys.to_vec()).unwrap();
                return Satisfiable::Yes(self_constraint);
            };
            assert_eq!(
                self.get_tags(keys),
                condition.get_tags(),
                "tag mismatch in TestPredicate::condition_on"
            );
            assert_eq!(
                keys,
                condition.required_bindings(),
                "Cannot be in same tag if keys are not the same"
            );

            if self == condition.predicate() {
                return Satisfiable::Tautology;
            }
            match self.get_tags(keys).into_iter().exactly_one().unwrap() {
                TestConstraintTag::One(_, _) => {
                    // predicates are mutually exclusive
                    Satisfiable::No
                }
                TestConstraintTag::Two(_, _) => {
                    if condition.predicate() == &TestPredicate::AlwaysTrueTwo {
                        // Does not teach us anything, leave unchanged
                        Satisfiable::Yes(self.clone().try_into_constraint(keys.to_vec()).unwrap())
                    } else {
                        Satisfiable::Tautology
                    }
                }
                TestConstraintTag::Three => {
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

    impl ConstraintTag<TestKey> for TestPredicate {
        type Tag = TestConstraintTag;

        fn get_tags(&self, keys: &[TestKey]) -> Vec<Self::Tag> {
            assert_eq!(self.arity(), keys.len());

            let args = keys.iter().cloned().collect_tuple();

            use TestPredicate::*;
            match self {
                AreEqualOne | NotEqualOne => {
                    let (a, b) = args.unwrap();
                    vec![TestConstraintTag::One(a, b)]
                }
                AreEqualTwo | AlwaysTrueTwo => {
                    let (a, b) = args.unwrap();
                    vec![TestConstraintTag::Two(a, b)]
                }
                NeverTrueThree | AlwaysTrueThree => vec![TestConstraintTag::Three],
            }
        }
    }

    impl Tag<TestKey, TestPredicate> for TestConstraintTag {
        type ExpansionFactor = u64;

        fn expansion_factor<'c, C>(
            &self,
            _constraints: impl IntoIterator<Item = C>,
        ) -> Self::ExpansionFactor
        where
            TestKey: 'c,
            TestPredicate: 'c,
            C: Into<(&'c TestPredicate, &'c [TestKey])>,
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
