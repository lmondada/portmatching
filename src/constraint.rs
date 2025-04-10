//! A [`Constraint`] type that can be used for pattern matching.
//!
//! This module provides an implementation of constraints that can be used
//! to create patterns and build pattern matchers.
//! It should cover most of the typical use cases for pattern matching.
//!
//! ## Predicates
//! Constraints are built from a predicate and a set of keys. Provided
//! bindings from keys to values, constraints can be evaluated to a boolean
//! by evaluating the predicate on the values. Users that wish to use
//! [`Constraint`] should define their own predicate type and implement
//! (all or some of) the traits [`ArityPredicate`], [`EvaluatePredicate`],
//! [`ConditionalPredicate`] and [`GetConstraintClass`].
//!
//! ## Patterns
//! Given an implementation of predicates, users can choose to use
//! [`ConstraintPattern`] to simplify the definition of patterns. These patterns
//! are defined by a set of constraints.
//!
//! ## Constraint selectors
//! The types [`DefaultConstraintEvaluator`] and [`DeterministicConstraintEvaluator`]
//! define simple constraint evaluators that can be provided to pattern matcher.

pub mod evaluator;
pub mod pattern;
pub mod predicate;
pub mod tag;

pub use evaluator::{
    ConstraintEvaluator, DefaultConstraintEvaluator, DeterministicConstraintEvaluator,
    EvaluateConstraints,
};
pub use pattern::{ConstraintPattern, PartialConstraintPattern};
pub use predicate::{ArityPredicate, ConditionalPredicate, EvaluatePredicate};
pub use tag::{ConstraintTag, Tag};

use std::{collections::BTreeSet, fmt::Debug};

use itertools::Itertools;
use thiserror::Error;

use crate::indexing::BindVariableError;

/// A set of constraints
pub type ConstraintSet<K, P> = BTreeSet<Constraint<K, P>>;

/// A constraint for pattern matching.
///
/// Given by a predicate of arity N and a vector of N arguments. Checking
/// that a constraint is satisfied is done in two steps:
/// 1. The keys in the arguments are resolved to values using index key-value
///    bindings.
/// 2. The predicate is then evaluated on the values.
///
/// ## Generic Parameters
/// - `K`: The index key type
/// - `P`: The predicate type
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Constraint<K, P> {
    predicate: P,
    args: Vec<K>,
}

/// Errors that occur when constructing constraints.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidConstraint {
    /// Cannot assign a value if the RHS is not a variable
    #[error("Cannot assign a value to {0}, it is not a variable")]
    AssignToValue(String),

    /// Invalid predicate arity
    #[error(
        "Mismatching arity: constraint expected arity {arguments_arity} but got predicate arity {predicate_arity}"
    )]
    InvalidArity {
        /// The arity of the predicate
        predicate_arity: usize,
        /// The actual arity of the arguments
        arguments_arity: usize,
    },

    /// Constraints refers to an unbound variable
    #[error("Constraint refered to unbound variable: {0}")]
    UnboundVariable(String),

    /// Bind a variable to a value that already exists
    #[error("Cannot bind variable {0}: already exists")]
    BindVariableExists(String),

    /// Bind a value to a variable that already exists
    #[error("Cannot bind value {value} to variable {variable}: value already exists")]
    BindValueExists {
        /// The value that already exists
        value: String,
        /// The variable binding the value to
        variable: String,
    },

    /// Bind a value to an unrecognised variable name
    #[error("Cannot bind to variable name: {0}")]
    InvalidVariableName(String),

    /// The predicate check failed, probably a malformed predicate
    #[error("Predicate check failed: {0}")]
    PredicateCheckFailed(String),
}

impl From<BindVariableError> for InvalidConstraint {
    fn from(e: BindVariableError) -> Self {
        match e {
            BindVariableError::VariableExists { key: var, .. } => {
                InvalidConstraint::BindVariableExists(var)
            }
            BindVariableError::InvalidKey { key: var, .. } => {
                InvalidConstraint::UnboundVariable(var)
            }
        }
    }
}

impl<K, P> Constraint<K, P> {
    /// Construct a binary constraint.
    ///
    /// Return an error if the number of arguments does not match the predicate
    /// arity.
    pub fn try_binary_from_triple(lhs: K, predicate: P, rhs: K) -> Result<Self, InvalidConstraint>
    where
        P: ArityPredicate,
    {
        Self::try_new(predicate, vec![lhs, rhs])
    }

    /// Construct a constraint.
    ///
    /// Return an error if the number of arguments does not match the predicate
    /// arity.
    pub fn try_new(predicate: P, args: Vec<K>) -> Result<Self, InvalidConstraint>
    where
        P: ArityPredicate,
    {
        if args.len() != predicate.arity() {
            Err(InvalidConstraint::InvalidArity {
                predicate_arity: predicate.arity(),
                arguments_arity: args.len(),
            })
        } else {
            Ok(Self { args, predicate })
        }
    }

    /// The arity of the constraint.
    pub fn arity(&self) -> usize
    where
        P: ArityPredicate,
    {
        assert_eq!(
            self.args.len(),
            self.predicate.arity(),
            "invalid constraint: arity mismatch"
        );
        self.args.len()
    }

    /// The bindings required to evaluate the constraint.
    pub fn required_bindings(&self) -> &[K] {
        &self.args
    }

    /// The constraint predicate
    pub fn predicate(&self) -> &P {
        &self.predicate
    }

    /// Convert the constraint to a tuple of predicate and arguments
    pub fn as_tuple_ref(&self) -> (&P, &[K]) {
        (&self.predicate, &self.args)
    }
}

impl<'c, K, P> From<&'c Constraint<K, P>> for (&'c P, &'c [K]) {
    fn from(constraint: &'c Constraint<K, P>) -> Self {
        constraint.as_tuple_ref()
    }
}

impl<K: Debug, P: Debug> Debug for Constraint<K, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.predicate)?;
        let args_str = self
            .args
            .iter()
            .map(|arg| format!("{:?}", arg))
            .collect_vec();
        write!(f, "({})", args_str.join(", "))?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    pub(crate) use super::predicate::tests::{
        TestConstraintTag, TestKey, TestPartialPattern, TestPattern, TestPredicate,
    };

    use crate::{indexing::tests::TestData, HashMap};

    use super::*;
    pub(crate) type TestConstraint = Constraint<TestKey, TestPredicate>;

    impl TestConstraint {
        pub(crate) fn new(pred: TestPredicate) -> TestConstraint {
            use TestPredicate::*;

            let key1 = "key1";
            let key2 = "key2";
            match pred {
                AreEqualOne | NotEqualOne | AreEqualTwo | AlwaysTrueTwo => {
                    TestConstraint::try_binary_from_triple(key1, pred, key2).unwrap()
                }
                NeverTrueThree | AlwaysTrueThree => {
                    TestConstraint::try_new(pred, vec![key1]).unwrap()
                }
            }
        }
    }

    #[test]
    fn test_construct_constraint_arity() {
        let c = TestConstraint::new(TestPredicate::AreEqualOne);
        assert_eq!(c.arity(), 2);

        assert_eq!(
            TestConstraint::try_binary_from_triple("yo", TestPredicate::AlwaysTrueThree, "lo",)
                .unwrap_err(),
            InvalidConstraint::InvalidArity {
                predicate_arity: 1,
                arguments_arity: 2,
            }
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_constraint() {
        let c = TestConstraint {
            predicate: TestPredicate::AreEqualOne,
            args: vec!["x"],
        };
        c.arity();
    }

    #[test]
    fn test_is_satisfied() {
        let c = TestConstraint::new(TestPredicate::AreEqualOne);
        let assmap = HashMap::from_iter([("key1", Some(1)), ("key2", Some(1)), ("key3", Some(3))]);
        let result = c.is_satisfied(&TestData, &assmap).unwrap();
        assert!(result);
    }

    #[test]
    fn test_not_is_satisfied() {
        let c = TestConstraint::new(TestPredicate::NotEqualOne);
        let assmap = HashMap::from_iter([("key1", Some(1)), ("key2", Some(1))]);
        let result = c.is_satisfied(&TestData, &assmap).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_not_bound() {
        let c = TestConstraint::new(TestPredicate::AreEqualOne);
        let assmap = HashMap::from_iter([("key1", Some(1)), ("key3", Some(2))]);
        let err_msg = c.is_satisfied(&TestData, &assmap).unwrap_err();
        assert_eq!(
            err_msg,
            InvalidConstraint::UnboundVariable(format!("{:?}", "key2".to_string()))
        );
    }
}
