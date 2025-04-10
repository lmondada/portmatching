//! Internal-only impl used by the det/default constraint evaluators.

use std::collections::BTreeSet;

use crate::indexing::IndexKey;

use crate::{ArityPredicate, ConstraintTag, EvaluatePredicate};

use super::{ConstraintEvaluator, EvaluateConstraints};

/// A helper struct to implement evaluators.
///
/// Both evaluators share the same implementation, but differ in the flag
/// `deterministic`. All public-facing implementations are delegated to this
/// struct.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct InnerEvaluator<K, P> {
    predicates: Vec<P>,
    all_required_bindings: Vec<K>,
    /// For each constraint, a list of indices into `all_required_bindings`
    /// required to evaluate it.
    binding_indices: Vec<Vec<usize>>,

    /// Whether the evaluators is deterministic (i.e. returns only the first or all
    /// constraints that match)
    deterministic: bool,
}

impl<K: IndexKey, P: Clone> InnerEvaluator<K, P> {
    pub(super) fn from_constraints<'c, C>(
        constraints: impl IntoIterator<Item = C>,
        deterministic: bool,
    ) -> Self
    where
        C: Into<(&'c P, &'c [K])>,
        P: 'c,
    {
        let transitions = constraints.into_iter();
        let size_hint = transitions.size_hint().0;

        let mut predicates = Vec::with_capacity(size_hint);
        let mut all_required_bindings = Vec::new();
        let mut binding_indices = Vec::with_capacity(size_hint);

        for constraint in transitions {
            let (pred, reqs) = constraint.into();

            let mut indices = Vec::new();
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
            predicates.push(pred.clone());
        }

        Self {
            predicates,
            all_required_bindings,
            binding_indices,
            deterministic,
        }
    }

    pub(super) fn keys(&self, pos: usize) -> Vec<K> {
        self.binding_indices[pos]
            .iter()
            .map(|&i| self.all_required_bindings[i])
            .collect()
    }

    pub(super) fn get_tag(&self) -> Option<P::Tag>
    where
        P: ConstraintTag<K> + ArityPredicate,
    {
        let fst_pred = self.predicates.first()?;
        let fst_keys = self.keys(0);
        let mut tags = BTreeSet::from_iter(fst_pred.get_tags(&fst_keys));

        for i in 1..self.predicates.len() {
            let pred = &self.predicates[i];
            let keys = self.keys(i);
            let new_tags = BTreeSet::from_iter(pred.get_tags(&keys));
            tags.retain(|cls| new_tags.contains(cls));
        }

        let Some(tag) = tags.pop_first() else {
            panic!("All predicates in a evaluator must share a tag")
        };

        Some(tag)
    }

    pub(super) fn predicates(&self) -> &[P] {
        &self.predicates
    }
}

impl<K, P> ConstraintEvaluator for InnerEvaluator<K, P>
where
    K: IndexKey,
{
    type Key = K;

    fn required_bindings(&self) -> &[Self::Key] {
        &self.all_required_bindings
    }
}

impl<K, P, Data, Value> EvaluateConstraints<Data, Value> for InnerEvaluator<K, P>
where
    P: EvaluatePredicate<Data, Value>,
    K: IndexKey,
{
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize> {
        let mut valid_pred = Vec::new();

        for i in 0..self.predicates.len() {
            if !valid_pred.is_empty() && self.deterministic {
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
