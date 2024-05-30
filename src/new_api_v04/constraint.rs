//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use super::{
    predicate::{AssignPredicate, FilterPredicate, Predicate},
    variable::{BindVariableError, VariableScope},
};
use std::fmt::Debug;
use thiserror::Error;

/// A literal for subject and object in constraints.
///
/// Literals are either a value from the predicate universe, or a variable
/// that will be bound at runtime to a value in the universe.
#[derive(Clone, Debug)]
pub enum ConstraintLiteral<V, U> {
    Variable(V),
    Value(U),
}

/// Errors that occur when evaluating literals.
#[derive(Clone, Debug, Error)]
pub enum LiteralEvalError {
    #[error("Unbound variable: {0}")]
    UnboundVariable(String),
}

/// Errors that occur when constructing constraints.
#[derive(Clone, Debug, Error)]
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
}

impl From<LiteralEvalError> for InvalidConstraint {
    fn from(e: LiteralEvalError) -> Self {
        match e {
            LiteralEvalError::UnboundVariable(var) => InvalidConstraint::UnboundVariable(var),
        }
    }
}

impl From<BindVariableError> for InvalidConstraint {
    fn from(e: BindVariableError) -> Self {
        match e {
            BindVariableError::VariableExists(var) => InvalidConstraint::BindVariableExists(var),
            BindVariableError::ValueExists { value, variable } => {
                InvalidConstraint::BindValueExists { value, variable }
            }
        }
    }
}

impl<V: Debug, U> ConstraintLiteral<V, U> {
    /// Evaluate a literal to a value in U.
    ///
    /// If the literal is a value, unwrap it. Otherwise, use the variable scope
    /// to resolve the binding.
    ///
    /// If the variable is not defined, this will panic.
    pub fn evaluate<'a>(
        &'a self,
        scope: &'a impl VariableScope<V, U>,
    ) -> Result<&'a U, LiteralEvalError> {
        match &self {
            ConstraintLiteral::Variable(var) => scope
                .get(var)
                .ok_or(LiteralEvalError::UnboundVariable(format!("{:?}", var))),
            ConstraintLiteral::Value(val) => Ok(val),
        }
    }
}

/// A constraint for pattern matching.
///
/// The following always holds:
/// - The number of arguments `args.len()` matches the predicate arity N.
/// - If the predicate is an `AssignPredicate`, the arity is >= 1 and the last
///   argument must be a variable.
///
/// Note that constraints must be carefully ordered when being satisfied as they make
/// assumptions on which variable bindings exist:
///  - For AssignPredicates, the first N-1 arguments must be constants or be
///    variables bound by previous constraints.
///  - For FilterPredicates, all arguments must be bound by previous constraints
///    if they are variables.
#[derive(Clone, Debug)]
pub struct Constraint<V, U, AP, FP> {
    predicate: Predicate<AP, FP>,
    args: Vec<ConstraintLiteral<V, U>>,
}

impl<V, U, AP, FP> Constraint<V, U, AP, FP>
where
    U: Debug,
    V: Debug,
    AP: AssignPredicate<U = U>,
    FP: FilterPredicate<U = U>,
{
    /// Construct a binary constraint.
    ///
    /// Returns an error if the constraint is malformed, i.e. if the predicate is
    /// a Predicate::Assign and the object is not a variable, or if the predicate
    /// is not binary.
    pub fn try_binary_from_triple(
        lhs: ConstraintLiteral<V, U>,
        predicate: Predicate<AP, FP>,
        rhs: ConstraintLiteral<V, U>,
    ) -> Result<Self, InvalidConstraint> {
        if predicate.arity() != 2 {
            return Err(InvalidConstraint::InvalidArity {
                predicate_arity: predicate.arity(),
                arguments_arity: 2,
            });
        }
        if matches!(predicate, Predicate::Assign(_)) && matches!(rhs, ConstraintLiteral::Value(_)) {
            return Err(InvalidConstraint::AssignToValue(format!("{:?}", rhs)));
        }
        Self::try_new(predicate, vec![lhs, rhs])
    }

    /// Construct a constraint.
    ///
    /// Returns an error if the number of arguments does not match the predicate
    /// arity.
    pub fn try_new(
        predicate: Predicate<AP, FP>,
        args: Vec<ConstraintLiteral<V, U>>,
    ) -> Result<Self, InvalidConstraint> {
        if args.len() != predicate.arity() {
            Err(InvalidConstraint::InvalidArity {
                predicate_arity: predicate.arity(),
                arguments_arity: args.len(),
            })
        } else if matches!(predicate, Predicate::Assign(_))
            && matches!(args.last(), Some(ConstraintLiteral::Value(_)))
        {
            Err(InvalidConstraint::AssignToValue(format!(
                "{:?}",
                args.last().unwrap()
            )))
        } else {
            Ok(Self { args, predicate })
        }
    }

    /// Return the arity of the constraint.
    pub fn arity(&self) -> usize {
        let arity = self.args.len();
        debug_assert_eq!(
            arity,
            self.predicate.arity(),
            "invalid constraint: arity mismatch"
        );
        arity
    }

    /// Return all variable assignments that would satisfy the constraint.
    ///
    /// This will panic if an `AssignPredicate` results in an invalid variable
    /// binding, or if the constraint is malformed.
    pub fn satisfy<S, D>(&self, data: &D, scope: S) -> Result<Vec<S>, InvalidConstraint>
    where
        S: Clone + VariableScope<V, U>,
        AP: AssignPredicate<D = D>,
        FP: FilterPredicate<D = D>,
        V: Clone,
    {
        let args = |n_args| {
            self.args[..n_args]
                .iter()
                .map(|arg| arg.evaluate(&scope))
                .collect::<Result<Vec<_>, _>>()
        };
        match &self.predicate {
            Predicate::Assign(ap) => {
                let lhs = ap.check_assign(data, &args(self.arity() - 1)?);
                let Some(ConstraintLiteral::Variable(rhs)) = self.args.last() else {
                    panic!("Invalid constraint: rhs of AssignPredicate is not a variable");
                };
                lhs.into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        scope.bind(rhs.clone(), obj)?;
                        Ok(scope)
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            Predicate::Filter(fp) => {
                let args = args(self.arity())?;
                if fp.check(data, &args) {
                    Ok([scope].to_vec())
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{new_api_v04::predicate::ArityPredicate, HashMap, HashSet};

    use super::*;

    #[derive(Debug)]
    struct AssignEq;
    impl ArityPredicate for AssignEq {
        fn arity(&self) -> usize {
            2
        }
    }

    impl AssignPredicate for AssignEq {
        type U = usize;
        type D = ();

        fn check_assign(&self, _: &Self::D, args: &[&Self::U]) -> HashSet<Self::U> {
            assert_eq!(args.len(), 1);
            HashSet::from_iter(args.iter().cloned().cloned())
        }
    }

    #[derive(Debug)]
    struct FilterEq;
    impl ArityPredicate for FilterEq {
        fn arity(&self) -> usize {
            2
        }
    }
    impl FilterPredicate for FilterEq {
        type U = usize;
        type D = ();

        fn check(&self, _: &Self::D, args: &[&Self::U]) -> bool {
            let [arg0, arg1] = args else {
                panic!("Invalid constraint: arity mismatch");
            };
            arg0 == arg1
        }
    }
    type TestConstraint = Constraint<String, usize, AssignEq, FilterEq>;

    #[test]
    fn test_construct_constraint() {
        let c = TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Assign(AssignEq),
            ConstraintLiteral::Variable("x".to_string()),
        )
        .unwrap();
        assert_eq!(c.arity(), 2);

        TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Assign(AssignEq),
            ConstraintLiteral::Value(1),
        )
        .expect_err("Invalid constraint: rhs of AssignPredicate is not a variable");

        TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Filter(FilterEq),
            ConstraintLiteral::Value(1),
        )
        .expect("A value RHS is valid for filters");

        TestConstraint::try_new(
            Predicate::Filter(FilterEq),
            vec![ConstraintLiteral::Value(1)],
        )
        .expect_err("Invalid constraint: arity mismatch");
    }

    #[test]
    fn test_assign_satisfy() {
        let c = TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Assign(AssignEq),
            ConstraintLiteral::Variable("x".to_string()),
        )
        .unwrap();
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x"), Some(&1));
    }

    #[test]
    fn test_filter_satisfy_value_to_value() {
        let c = TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Filter(FilterEq),
            ConstraintLiteral::Value(1),
        )
        .unwrap();
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_filter_satisfy_value_to_different_value() {
        let c = TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Filter(FilterEq),
            ConstraintLiteral::Value(2),
        )
        .unwrap();
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_assign_satisfy_value_to_variable() {
        let c = TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Value(1),
            Predicate::Assign(AssignEq),
            ConstraintLiteral::Variable("y".to_string()),
        )
        .unwrap();
        let scope = HashMap::from_iter([("x".to_string(), 1)]);
        let result = c.satisfy(&(), scope).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("y"), Some(&1));
    }
}
