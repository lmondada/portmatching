use crate::{
    indexing::IndexKey, ArityPredicate, ConstraintEvaluator, ConstraintTag, EvaluateConstraints,
    EvaluatePredicate,
};

use super::InnerEvaluator;

use delegate::delegate;

/// A default evaluator for predicate-based patterns.
///
/// This evaluator is non-deterministic, meaning it will return all constraints
/// that match.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DefaultConstraintEvaluator<K, P>(InnerEvaluator<K, P>);

impl<K: IndexKey, P: Clone> DefaultConstraintEvaluator<K, P> {
    /// Create a new evaluator from a set of constraints.
    ///
    /// The constraints are used to initialize the evaluator.
    pub fn from_constraints<'c>(constraints: impl IntoIterator<Item = (&'c P, &'c [K])>) -> Self
    where
        K: 'c,
        P: 'c,
    {
        Self(InnerEvaluator::from_constraints(constraints, false))
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

impl<K, P> ConstraintEvaluator for DefaultConstraintEvaluator<K, P>
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

impl<K, P, Data, Value> EvaluateConstraints<Data, Value> for DefaultConstraintEvaluator<K, P>
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
