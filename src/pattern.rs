//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use thiserror::Error;

/// A pattern for pattern matching.
///
/// Define valid patterns by providing a conversion to a vector of constraints.
pub trait Pattern {
    /// The constraint type that the pattern is defined over.
    type Constraint;
    /// The error type returned by the conversion to a vector of constraints.
    type Error;

    /// Convert to a vector of constraints.
    fn try_to_constraint_vec(&self) -> Result<Vec<Self::Constraint>, Self::Error>;
}

/// A pattern that can be viewed as concrete data that can be matched on.
///
/// This excludes patterns that may map on multiple concrete hosts (e.g.
/// patterns with wildcards, variable number of nodes, etc.)
pub trait ConcretePattern: Pattern {
    /// The concrete host data that the pattern maps on.
    type Host;

    /// View pattern as host data that can be matched on.
    fn as_host(&self) -> &Self::Host;
}

/// A concrete pattern was expected but failed at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("Pattern is not concrete and cannot be matched directly")]
pub struct NonConcretePattern;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use derive_more::{From, Into};

    #[derive(Debug, Clone, PartialEq, Eq, Hash, From, Into)]
    pub(crate) struct TestPattern<C>(Vec<C>);
    impl<C: Clone> Pattern for TestPattern<C> {
        type Constraint = C;
        type Error = ();

        fn try_to_constraint_vec(&self) -> Result<Vec<Self::Constraint>, Self::Error> {
            Ok(self.0.clone())
        }
    }
}
