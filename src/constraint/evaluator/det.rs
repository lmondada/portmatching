use crate::{
    indexing::IndexKey, ArityPredicate, ConstraintEvaluator, ConstraintTag, EvaluateConstraints,
    EvaluatePredicate,
};

use super::InnerEvaluator;

use delegate::delegate;

/// A deterministic evaluator for predicate-based patterns.
///
/// This evaluator will return only the first constraint that matches.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeterministicConstraintEvaluator<K, P>(InnerEvaluator<K, P>);

impl<K: IndexKey, P: Clone> DeterministicConstraintEvaluator<K, P> {
    /// Create a new evaluator from a set of constraints.
    pub fn from_constraints<'c, C>(constraints: impl IntoIterator<Item = C>) -> Self
    where
        C: Into<(&'c P, &'c [K])>,
        P: 'c,
    {
        Self(InnerEvaluator::from_constraints(constraints, true))
    }

    delegate! {
        to self.0 {
            /// Get the keys at a given position.
            ///
            /// The position is used to index into the `binding_indices` vector.
            pub fn keys(&self, pos: usize) -> Vec<K>;

            /// Get the constraint tag for this evaluator.
            ///
            /// The constraint tag is determined by the predicates in the evaluator.
            pub fn get_tag(&self) -> Option<P::Tag>
            where
                P: ConstraintTag<K> + ArityPredicate;

            /// Get the predicates in this evaluator.
            ///
            /// The predicates are stored in the `predicates` vector.
            pub fn predicates(&self) -> &[P];
        }
    }
}

impl<K, P> ConstraintEvaluator for DeterministicConstraintEvaluator<K, P>
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

impl<K, P, Data, Value> EvaluateConstraints<Data, Value> for DeterministicConstraintEvaluator<K, P>
where
    P: EvaluatePredicate<Data, Value>,
    K: IndexKey,
{
    delegate! {
        to self.0 {
            fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
        }
    }
}
