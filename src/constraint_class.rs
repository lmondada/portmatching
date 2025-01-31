//! Group constraints together into constraint classes.

/// Type for expansion factors.
///
/// This would most naturally be a floating point number, but we use an integer
/// to have total ordering (and avoid floating point arithmetic).
///
/// Multiply all expansion factors by a constant amount (eg 10eX) and round
/// to nearest integer. As long as the factor is the same across all constraints,
/// then it does not matter what the constant is.
pub type ExpansionFactor = u64;

/// Constraint classes group constraints together that are related to one
/// another. You would typically put constraints in the same group if they
/// are mutually exclusive, or have some other logical relationship---or simply
/// because evaluating them jointly is more efficient.
///
/// You can think of this grouping as a partition of your constraints into
/// sets such that constraints of two different sets are independent of each
/// other. Except we do not enforce independence (you just won't be able to
/// make use of that dependency), and the sets do not need to be disjoint.
pub trait ConstraintClass<C>: Sized + Ord {
    /// Get all classes that the constraint belongs to.
    fn get_classes(constraint: &C) -> Vec<Self>;

    /// Compute the expansion factor of the constraints.
    ///
    /// The expansion factor is the relative change in the expected number of
    /// bindings after the constraints are applied. When applying constraints,
    /// only the bindings that satisfy the constraint are retained, so typically
    /// the expansion factor is less than 1. However, it could be that more
    /// than one constraint is satisfied, in which case the same binding might
    /// be retained multiple times, leading to greater factors.
    ///
    /// It is guaranteed that all constraints in the input are in the class
    /// `self`.
    fn expansion_factor<'c>(&self, constraints: impl IntoIterator<Item = &'c C>) -> ExpansionFactor
    where
        C: 'c;
}
