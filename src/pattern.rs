//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use thiserror::Error;

use crate::Constraint;

/// A pattern for pattern matching.
///
/// Define valid patterns by providing a conversion to a vector of constraints.
pub trait Pattern {
    /// The type of variable names used in the pattern.
    type Key;
    /// The type of predicates used in the pattern.
    type Predicate;
    /// The error type returned by the conversion to a vector of constraints.
    type Error;

    /// Convert to a vector of constraints.
    fn try_to_constraint_vec(
        &self,
    ) -> Result<Vec<Constraint<Self::Key, Self::Predicate>>, Self::Error>;

    /// Optionally, get a list of required bindings.
    ///
    /// The set of bindings used during matching will always be returned (and
    /// used by default if no further bindings are given).
    fn required_bindings(&self) -> Option<Vec<Self::Key>> {
        None
    }
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
    pub(crate) struct TestPattern<K, P>(Vec<Constraint<K, P>>);
    impl<K: Clone, P: Clone> Pattern for TestPattern<K, P> {
        type Key = K;
        type Predicate = P;
        type Error = ();

        fn try_to_constraint_vec(&self) -> Result<Vec<Constraint<K, P>>, Self::Error> {
            Ok(self.0.clone())
        }
    }
}
