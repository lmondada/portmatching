//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use super::{
    predicate::{AssignPredicate, FilterPredicate, Predicate},
    variable::VariableScope,
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
    #[error("Cannot assign a value if the RHS is not a variable")]
    AssignToValue,

    /// Invalid predicate arity
    #[error("Mismatching predicate arity and argument length")]
    InvalidArity,

    /// Constraints refers to an unbound variable
    #[error("Constraint refered to unbound variable: {0}")]
    UnboundVariable(String),
}

impl From<LiteralEvalError> for InvalidConstraint {
    fn from(e: LiteralEvalError) -> Self {
        match e {
            LiteralEvalError::UnboundVariable(var) => InvalidConstraint::UnboundVariable(var),
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
            return Err(InvalidConstraint::InvalidArity);
        }
        if matches!(predicate, Predicate::Assign(_)) && matches!(rhs, ConstraintLiteral::Value(_)) {
            return Err(InvalidConstraint::AssignToValue);
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
            Err(InvalidConstraint::InvalidArity)
        } else if matches!(predicate, Predicate::Assign(_))
            && matches!(args.last(), Some(ConstraintLiteral::Value(_)))
        {
            Err(InvalidConstraint::AssignToValue)
        } else {
            Ok(Self { args, predicate })
        }
    }

    /// Return the arity of the constraint.
    pub fn arity(&self) -> usize {
        let arity = self.args.len();
        debug_assert_eq!(arity, self.predicate.arity(), "invalid constraint: arity mismatch");
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
                let Some(ConstraintLiteral::Variable(rhs)) = &self.args.last() else {
                    panic!("Invalid constraint: rhs of AssignPredicate is not a variable");
                };
                Ok(lhs
                    .into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        scope.bind(rhs, obj).unwrap();
                        scope
                    })
                    .collect())
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
