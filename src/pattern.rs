/// A pattern for pattern matching.
///
/// A pattern is defined by
///  - a transformation into a vector of constraints
///  - a pattern root
///  - a way to express a pattern as the underlying graph
pub trait Pattern {
    type Constraint;
    type Host;
    type U;

    /// Convert to a vector of constraints.
    fn to_constraint_vec(&self) -> Vec<Self::Constraint>;

    /// Get pattern viewed as a host data that can be matched on.
    fn as_host(&self) -> &Self::Host;

    /// Get the root of the pattern.
    fn root(&self) -> Self::U;
}
