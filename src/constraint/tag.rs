//! Constraint tags group constraints together that are related to one
//! another. You would typically give constraints the same tag if they
//! are mutually exclusive, or have some other logical relationship---or simply
//! because evaluating them jointly is more efficient.
//!
//! You can think of this grouping as a partition of your constraints into
//! sets such that constraints of two different sets are independent of each
//! other. Except we do not enforce independence (you just won't be able to
//! make use of that dependency), and the sets do not need to be disjoint.

use super::ConstraintEvaluator;

/// Implement on a predicate to define the constraint tags that constraints
/// belongs to.
pub trait ConstraintTag<K>: Sized {
    /// Sets of constraints that can be evaluated together form branch tags.
    type Tag: Tag<K, Self>;

    /// Get all tags of the constraint with predicate `self` and keys `keys`.
    ///
    /// # Panics
    /// This may panic if the keys are not of the right arity and type.
    fn get_tags(&self, keys: &[K]) -> Vec<Self::Tag>;
}

/// A tag for a constraint type.
pub trait Tag<K, P>: Ord + std::fmt::Debug {
    /// Type for expansion factors.
    ///
    /// This would most naturally be a floating point number, but should have a
    /// total ordering. This could be the "true" expansion factor multiplied
    /// by a constant amount (eg 10eX) and rounded to the nearest integer. As
    /// long as the factor is the same across all constraints, then it does
    /// not matter what the constant is.
    type ExpansionFactor: Ord;

    /// The type of constraint evaluator for this tag.
    type Evaluator: ConstraintEvaluator<Key = K>;

    /// Compute the expansion factor of the constraints.
    ///
    /// The expansion factor is the relative change in the expected number of
    /// bindings after the constraints are applied. When applying constraints,
    /// only the bindings that satisfy the constraint are retained, so typically
    /// the expansion factor is less than 1. However, it could be that more
    /// than one constraint is satisfied, in which case the same binding might
    /// be retained multiple times, leading to greater factors.
    ///
    /// It is guaranteed that all constraints in the input have the tag `self`.
    fn expansion_factor<'c, C>(
        &self,
        constraints: impl IntoIterator<Item = C>,
    ) -> Self::ExpansionFactor
    where
        K: 'c,
        P: 'c,
        C: Into<(&'c P, &'c [K])>;

    /// Construct an evaluator for the list of constraints.
    ///
    /// All constraints are guaranteed to have the tag `self`.
    fn compile_evaluator<'c, C>(&self, constraints: impl IntoIterator<Item = C>) -> Self::Evaluator
    where
        K: 'c,
        P: 'c,
        C: Into<(&'c P, &'c [K])>;
}
