//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use crate::{
    indexing::{Binding, IndexKey, IndexedData},
    predicate::ArityPredicate,
};

use super::{
    indexing::{BindMap, BindVariableError},
    predicate::Predicate,
};
use itertools::Itertools;
use std::{borrow::Borrow, cell::RefCell, collections::BTreeSet, fmt::Debug};
use thiserror::Error;

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

/// A heuristic whether a set of constraints should be turned into a deterministic
/// transition.
///
/// More deterministic states will result in faster automaton runtimes, but at
/// the cost of larger automata. A good heuristic is therefore important to
/// find the best tradeoff.
#[derive(Default)]
pub enum DetHeuristic<K, P> {
    /// Turn into deterministic state whenever indicated by the constraint tree
    #[default]
    Default,
    /// Never turn into deterministic state
    Never,
    /// Use a custom heuristic that is ANDed with the constraint tree make_det flag
    #[allow(clippy::type_complexity)]
    Custom(RefCell<Box<dyn FnMut(&[&Constraint<K, P>]) -> bool>>),
}

impl<K, P> DetHeuristic<K, P> {
    /// Whether the constraints should be turned into a deterministic transition.
    ///
    /// This cannot override the `make_det` flag in the constraint tree.
    pub fn make_det(&self, constraints: &[&Constraint<K, P>]) -> bool {
        match self {
            DetHeuristic::Default => true,
            DetHeuristic::Never => false,
            DetHeuristic::Custom(f) => f.borrow_mut()(constraints),
        }
    }
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

    /// Evaluate the constraint given the subject data and index map.
    ///
    /// # Arguments
    ///
    /// * `data` - The data against which the constraint is evaluated
    /// * `known_bindings` - The current index map containing key-value bindings
    ///
    /// # Returns
    ///
    /// `Result<bool, InvalidConstraint>` - Ok(true) if the constraint is satisfied,
    /// Ok(false) if it's not, or an Err if there's an invalid constraint.
    pub fn is_satisfied<D: IndexedData<K>>(
        &self,
        data: &D,
        known_bindings: &D::BindMap,
    ) -> Result<bool, InvalidConstraint>
    where
        P: Predicate<D, D::Value>,
        K: IndexKey,
    {
        let args = self
            .args
            .iter()
            .map(|key| {
                let Binding::Bound(value) = known_bindings.get_binding(key) else {
                    return Err(InvalidConstraint::UnboundVariable(format!("{key:?}")));
                };
                Ok(value.borrow().clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.predicate.check(&args, data))
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
    use crate::{
        indexing::tests::TestData,
        predicate::tests::{TestKey, TestPredicate},
        HashMap,
    };

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
