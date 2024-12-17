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
use std::{collections::HashSet, hash::Hash};

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
        P::BranchClass: Hash,
        K: Ord,
    {
        let mut constraints_set = ConstraintSet::default();
        let mut known_classes = HashSet::<P::BranchClass>::default();

        for c in constraints {
            if !known_classes.insert(c.get_class()) {
                panic!("Cannot have two constraints with the same class in a pattern");
            }
            constraints_set.insert(c);
        }

        Self {
            constraints: constraints_set,
        }
    }

    pub fn all_classes(&self) -> impl Iterator<Item = P::BranchClass> + '_ {
        self.constraints.iter().map(|p| p.get_class())
    }

    pub fn get_by_class(&self, cls: &P::BranchClass) -> Option<&Constraint<K, P>> {
        self.constraints.iter().find(|p| &p.get_class() == cls)
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

    fn get_branch_classes(&self) -> impl Iterator<Item = (Self::BranchClass, ClassRank)> {
        self.all_classes().map(|cls| (cls, 0.5))
    }

    fn condition_on<'p>(
        &self,
        cls: &Self::BranchClass,
        conditions: impl IntoIterator<Item = &'p Self::Constraint>,
    ) -> impl Iterator<Item = (Option<Self::Constraint>, Self)>
    where
        Self: 'p,
    {
        let mut conditions_map = HashMap::default();

        let cls_constraint = self.get_by_class(cls).cloned();
        if let Some(c) = cls_constraint.clone() {
            conditions_map.insert(cls.clone(), c);
        }

        for c in conditions {
            if conditions_map.insert(c.get_class(), c.clone()).is_some() {
                unimplemented!("Multiple constraints with the same class in condition_on");
            }
        }

        let mut new_constraints = Vec::new();

        for c in self.constraints.iter() {
            if let Some(constraint) = conditions_map.get(&c.get_class()) {
                let new_c = match c.condition_on(constraint) {
                    Satisfiable::Yes(new_c) => Some(new_c),
                    Satisfiable::Tautology => None,
                    Satisfiable::No => {
                        // Return an empty iterator
                        return None.into_iter();
                    }
                };
                if let Some(new_c) = new_c {
                    new_constraints.push(new_c);
                }
            } else {
                new_constraints.push(c.clone());
            }
        }

        assert!(
            cls_constraint.is_none() || new_constraints.iter().all(|c| &c.get_class() != cls),
            "The conditioned results still include constraints of the original class."
        );

        Some((cls_constraint, Self::from_constraints(new_constraints))).into_iter()
    }

    fn is_satisfiable(&self) -> Satisfiable {
        if self.constraints.is_empty() {
            Satisfiable::Tautology
        } else {
            Satisfiable::Yes(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicatePatternDefaultSelector<K, P> {
    predicates: Vec<P>,
    all_required_bindings: Vec<K>,
    /// For each constraint, a list of indices into `all_required_bindings`
    /// required to evaluate it.
    binding_indices: Vec<Vec<usize>>,
}

impl<K: IndexKey, P: Clone> PredicatePatternDefaultSelector<K, P> {
    pub fn from_constraints<'c>(constraints: impl IntoIterator<Item = &'c Constraint<K, P>>) -> Self
    where
        Constraint<K, P>: 'c,
    {
        let constraints = constraints.into_iter();
        let n_constraints = constraints.size_hint().0;

        let mut predicates = Vec::with_capacity(n_constraints);
        let mut all_required_bindings = Vec::new();
        let mut binding_indices = Vec::with_capacity(n_constraints);

        for c in constraints {
            let reqs = c.required_bindings();
            let mut indices = Vec::with_capacity(reqs.len());

            for &req in reqs {
                let pos = all_required_bindings.iter().position(|&k| k == req);
                if let Some(pos) = pos {
                    indices.push(pos);
                } else {
                    all_required_bindings.push(req);
                    indices.push(all_required_bindings.len() - 1);
                }
            }
            binding_indices.push(indices);
            predicates.push(c.predicate().clone());
        }

        Self {
            predicates,
            all_required_bindings,
            binding_indices,
        }
    }

    pub fn keys(&self, pos: usize) -> Vec<K> {
        self.binding_indices[pos]
            .iter()
            .map(|&i| self.all_required_bindings[i])
            .collect()
    }

    pub fn get_class(&self) -> Option<P::BranchClass>
    where
        P: ConstraintLogic<K>,
    {
        let fst_pred = self.predicates.first()?;
        let fst_keys = self.keys(0);
        let cls = fst_pred.get_class(&fst_keys);

        for i in 1..self.predicates.len() {
            let keys = self.keys(i);
            let pred = &self.predicates[i];
            if cls != pred.get_class(&keys) {
                panic!("All predicates in a pattern must have the same class")
            }
        }

        Some(cls)
    }

    pub fn predicates(&self) -> &[P] {
        &self.predicates
    }
}

impl<K, P> BranchSelector for PredicatePatternDefaultSelector<K, P>
where
    K: IndexKey,
{
    type Key = K;

    fn required_bindings(&self) -> &[Self::Key] {
        &self.all_required_bindings
    }
}

impl<K, P, Data, Value> EvaluateBranchSelector<Data, Value>
    for PredicatePatternDefaultSelector<K, P>
where
    P: Predicate<Data, Value>,
    K: IndexKey,
{
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize> {
        let mut valid_pred = Vec::new();

        for i in 0..self.predicates.len() {
            let constraint = &self.predicates[i];
            let indices = &self.binding_indices[i];

            let Ok(bindings) = indices
                .iter()
                .map(|&i| bindings[i].as_ref().ok_or(()))
                .collect::<Result<Vec<_>, _>>()
            else {
                continue;
            };
            if constraint.check(&bindings, data) {
                valid_pred.push(i);
            }
        }

        valid_pred
    }
}

impl<K: IndexKey, P: Clone> CreateBranchSelector<Constraint<K, P>>
    for PredicatePatternDefaultSelector<K, P>
{
    fn create_branch_selector(constraints: Vec<Constraint<K, P>>) -> Self {
        Self::from_constraints(&constraints)
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
