use std::collections::BTreeSet;

use delegate::delegate;

use crate::{
    branch_selector::{BranchSelector, CreateBranchSelector, EvaluateBranchSelector},
    indexing::IndexKey,
    Constraint,
};

use super::{ConstraintLogic, Predicate};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PredicatePatternDefaultSelector<K, P>(InnerSelector<K, P>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeterministicPredicatePatternSelector<K, P>(InnerSelector<K, P>);

/// A helper struct to implement [`PredicatePatternDefaultSelector`] and
/// [`DeterministicPredicatePatternSelector`].
///
/// Both selectors share the same implementation, but differ in the flag
/// `deterministic`. All public-facing implementations are delegated to this
/// struct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct InnerSelector<K, P> {
    predicates: Vec<P>,
    all_required_bindings: Vec<K>,
    /// For each constraint, a list of indices into `all_required_bindings`
    /// required to evaluate it.
    binding_indices: Vec<Vec<usize>>,

    /// Whether the selector is deterministic (i.e. returns only the first or all
    /// constraints that match)
    deterministic: bool,
}

impl<K: IndexKey, P: Clone> PredicatePatternDefaultSelector<K, P> {
    pub fn from_constraints<'c>(constraints: impl IntoIterator<Item = &'c Constraint<K, P>>) -> Self
    where
        Constraint<K, P>: 'c,
    {
        Self(InnerSelector::from_constraints(constraints, false))
    }

    delegate! {
        to self.0 {
            pub fn keys(&self, pos: usize) -> Vec<K>;

            pub fn get_class(&self) -> Option<P::BranchClass>
            where
                P: ConstraintLogic<K>;

            pub fn predicates(&self) -> &[P];
        }
    }
}

impl<K: IndexKey, P: Clone> DeterministicPredicatePatternSelector<K, P> {
    pub fn from_constraints<'c>(constraints: impl IntoIterator<Item = &'c Constraint<K, P>>) -> Self
    where
        Constraint<K, P>: 'c,
    {
        Self(InnerSelector::from_constraints(constraints, true))
    }

    delegate! {
        to self.0 {
            pub fn keys(&self, pos: usize) -> Vec<K>;

            pub fn get_class(&self) -> Option<P::BranchClass>
            where
                P: ConstraintLogic<K>;

            pub fn predicates(&self) -> &[P];
        }
    }
}

impl<K: IndexKey, P: Clone> CreateBranchSelector<Constraint<K, P>>
    for PredicatePatternDefaultSelector<K, P>
{
    fn create_branch_selector(constraints: Vec<Constraint<K, P>>) -> Self {
        Self::from_constraints(&constraints)
    }
}

impl<K: IndexKey, P: Clone> CreateBranchSelector<Constraint<K, P>>
    for DeterministicPredicatePatternSelector<K, P>
{
    fn create_branch_selector(constraints: Vec<Constraint<K, P>>) -> Self {
        Self::from_constraints(&constraints)
    }
}

impl<K, P> BranchSelector for PredicatePatternDefaultSelector<K, P>
where
    K: IndexKey,
{
    type Key = K;

    delegate! {
        to self.0 {
            fn required_bindings(&self) -> &[Self::Key];
        }
    }
}

impl<K, P> BranchSelector for DeterministicPredicatePatternSelector<K, P>
where
    K: IndexKey,
{
    type Key = K;

    delegate! {
        to self.0 {
            fn required_bindings(&self) -> &[Self::Key];
        }
    }
}

impl<K, P, Data, Value> EvaluateBranchSelector<Data, Value>
    for PredicatePatternDefaultSelector<K, P>
where
    P: Predicate<Data, Value>,
    K: IndexKey,
{
    delegate! {
        to self.0 {
            fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
        }
    }
}

impl<K, P, Data, Value> EvaluateBranchSelector<Data, Value>
    for DeterministicPredicatePatternSelector<K, P>
where
    P: Predicate<Data, Value>,
    K: IndexKey,
{
    delegate! {
        to self.0 {
            fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
        }
    }
}

impl<K: IndexKey, P: Clone> InnerSelector<K, P> {
    fn from_constraints<'c>(
        constraints: impl IntoIterator<Item = &'c Constraint<K, P>>,
        deterministic: bool,
    ) -> Self
    where
        Constraint<K, P>: 'c,
    {
        let transitions = constraints.into_iter();
        let size_hint = transitions.size_hint().0;

        let mut predicates = Vec::with_capacity(size_hint);
        let mut all_required_bindings = Vec::new();
        let mut binding_indices = Vec::with_capacity(size_hint);

        for constraint in transitions {
            let pred = constraint.predicate().clone();

            let mut indices = Vec::new();
            let reqs = constraint.required_bindings();
            indices.reserve(reqs.len());

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
            predicates.push(pred);
        }

        Self {
            predicates,
            all_required_bindings,
            binding_indices,
            deterministic,
        }
    }

    fn keys(&self, pos: usize) -> Vec<K> {
        self.binding_indices[pos]
            .iter()
            .map(|&i| self.all_required_bindings[i])
            .collect()
    }

    fn get_class(&self) -> Option<P::BranchClass>
    where
        P: ConstraintLogic<K>,
    {
        let fst_pred = self.predicates.first()?;
        let fst_keys = self.keys(0);
        let mut classes = BTreeSet::from_iter(fst_pred.get_classes(&fst_keys));

        for i in 1..self.predicates.len() {
            let pred = &self.predicates[i];
            let keys = self.keys(i);
            let new_classes = BTreeSet::from_iter(pred.get_classes(&keys));
            classes.retain(|cls| new_classes.contains(cls));
        }

        let Some(cls) = classes.pop_first() else {
            panic!("All predicates in a pattern must share a class")
        };

        Some(cls)
    }

    fn predicates(&self) -> &[P] {
        &self.predicates
    }
}

impl<K, P> BranchSelector for InnerSelector<K, P>
where
    K: IndexKey,
{
    type Key = K;

    fn required_bindings(&self) -> &[Self::Key] {
        &self.all_required_bindings
    }
}

impl<K, P, Data, Value> EvaluateBranchSelector<Data, Value> for InnerSelector<K, P>
where
    P: Predicate<Data, Value>,
    K: IndexKey,
{
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize> {
        let mut valid_pred = Vec::new();

        for i in 0..self.predicates.len() {
            if valid_pred.len() > 0 && self.deterministic {
                // Stop at the first valid predicate
                break;
            }

            let predicate = &self.predicates[i];
            let indices = &self.binding_indices[i];

            let Ok(bindings) = indices
                .iter()
                .map(|&i| bindings[i].as_ref().ok_or(()))
                .collect::<Result<Vec<_>, _>>()
            else {
                continue;
            };

            predicate.check(&bindings, data).then(|| valid_pred.push(i));
        }

        valid_pred
    }
}
