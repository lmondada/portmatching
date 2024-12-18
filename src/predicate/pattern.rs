//! Implement [`crate::Pattern`] traits by defining predicates and their logic.

use derive_more::From;
use itertools::Itertools;

use crate::{
    branch_selector::{BranchSelector, CreateBranchSelector, EvaluateBranchSelector},
    constraint::ConstraintSet,
    indexing::IndexKey,
    pattern::{ClassRank, Pattern, Satisfiable},
    Constraint, HashMap, PatternLogic, Predicate,
};
use core::panic;
use std::{
    collections::{BTreeSet, HashSet},
    hash::Hash,
};

use super::ConstraintLogic;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicateLogic<K, P> {
    constraints: ConstraintSet<K, P>,
}

impl<K, P> PredicateLogic<K, P>
where
    P: ConstraintLogic<K>,
{
    /// Create a predicate pattern from a set of constraints
    ///
    /// Every constraint must have a different class, otherwise this will panic.
    pub fn from_constraints(constraints: impl IntoIterator<Item = Constraint<K, P>>) -> Self
    where
        K: Ord,
    {
        Self::from_constraints_set(constraints.into_iter().collect())
    }

    pub fn from_constraints_set(constraints: BTreeSet<Constraint<K, P>>) -> Self {
        Self { constraints }
    }

    pub fn all_classes(&self) -> BTreeSet<P::BranchClass> {
        self.constraints
            .iter()
            .flat_map(|p| p.get_classes())
            .collect()
    }

    pub fn get_by_class<'c>(
        &'c self,
        cls: &'c P::BranchClass,
    ) -> impl Iterator<Item = &'c Constraint<K, P>> {
        self.constraints
            .iter()
            .filter(|p| p.get_classes().contains(cls))
    }

    pub fn conditioned(
        &self,
        known_constraints: &BTreeSet<Constraint<K, P>>,
        prev_constraints: &[Constraint<K, P>],
    ) -> Satisfiable<Self>
    where
        K: IndexKey,
    {
        let mut new_constraints = BTreeSet::new();

        for c in &self.constraints {
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

        Satisfiable::Yes(Self::from_constraints(new_constraints))
    }
}

/// A dumb wrapper around [`PredicateLogic`] that implements [`Pattern`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, From)]
pub struct PredicatePattern<K, P>(PredicateLogic<K, P>);

impl<K, P> PredicatePattern<K, P>
where
    P: ConstraintLogic<K>,
    P::BranchClass: Hash,
    K: Ord,
{
    pub fn from_constraints(constraints: impl IntoIterator<Item = Constraint<K, P>>) -> Self {
        Self(PredicateLogic::from_constraints(constraints))
    }

    pub fn from_constraints_set(constraints: BTreeSet<Constraint<K, P>>) -> Self {
        Self(PredicateLogic::from_constraints_set(constraints))
    }
}

impl<K, P> Pattern for PredicatePattern<K, P>
where
    K: IndexKey,
    P: ConstraintLogic<K>,
    P::BranchClass: Hash + Clone,
{
    type Key = K;

    type Logic = PredicateLogic<K, P>;

    type Constraint = Constraint<K, P>;

    fn required_bindings(&self) -> Vec<Self::Key> {
        self.0
            .constraints
            .iter()
            .flat_map(|p| p.required_bindings().iter().copied())
            .unique()
            .collect()
    }

    fn into_logic(self) -> Self::Logic {
        self.0
    }
}

impl<K, P> PatternLogic for PredicateLogic<K, P>
where
    K: IndexKey,
    P: ConstraintLogic<K>,
    P::BranchClass: Hash + Clone,
{
    type Constraint = Constraint<K, P>;

    type BranchClass = P::BranchClass;

    fn rank_classes(&self) -> impl Iterator<Item = (Self::BranchClass, ClassRank)> {
        self.all_classes().into_iter().map(|cls| (cls, 0.5))
    }

    fn nominate(&self, cls: &Self::BranchClass) -> BTreeSet<Self::Constraint> {
        self.constraints
            .iter()
            .filter(|c| c.get_classes().iter().any(|c| c == cls))
            .cloned()
            .collect()
    }

    fn condition_on(
        &self,
        constraints: &[Self::Constraint],
        known_constraints: &BTreeSet<Self::Constraint>,
    ) -> Vec<Satisfiable<Self>> {
        let mut conditioned = Vec::with_capacity(constraints.len());

        for (i, c) in constraints.iter().enumerate() {
            let mut known_constraints = known_constraints.clone();
            known_constraints.insert(c.clone());
            let prev_constraints = &constraints[..i];
            conditioned.push(self.conditioned(&known_constraints, prev_constraints));
        }

        conditioned
    }

    fn is_satisfiable(&self) -> Satisfiable {
        if self.constraints.is_empty() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constraint::tests::TestConstraint,
        pattern::Pattern,
        predicate::tests::{TestPattern, TestPredicate},
    };

    #[test]
    fn test_required_bindings() {
        let c = TestConstraint::try_binary_from_triple("key1", TestPredicate::AreEqualOne, "key1")
            .unwrap();
        let p = TestPattern::from_constraints(vec![c]);

        assert_eq!(p.required_bindings(), vec!["key1"]);
    }
}
