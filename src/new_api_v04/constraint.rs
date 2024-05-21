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
/// Constraints must be carefully ordered when being satisfied as they make
/// assumptions on which variable bindings exist:
///  - For AssignPredicates, the lhs, if it is a variable, must be already bound,
///    and the rhs must always a variable.
///  - For FilterPredicates, both lhs and rhs must be bound if they are variables.
#[derive(Clone, Debug)]
pub struct Constraint<V, U, AP, FP> {
    lhs: ConstraintLiteral<V, U>,
    predicate: Predicate<AP, FP>,
    rhs: ConstraintLiteral<V, U>,
}

/// Errors that occur when constructing constraints.
#[derive(Clone, Debug, Error)]
pub enum InvalidConstraint {
    /// Cannot assign a value if the RHS is not a variable
    #[error("Cannot assign a value if the RHS is not a variable")]
    AssignToValue,
}

impl<V, U, AP, FP> Constraint<V, U, AP, FP>
where
    V: Debug,
    AP: AssignPredicate<U = U>,
    FP: FilterPredicate<U = U>,
{
    /// Construct a constraint.
    ///
    /// Returns an error if the constraint is malformed, i.e. if the predicate is
    /// a Predicate::Assign and the object is not a variable.
    pub fn try_from_triple(
        lhs: ConstraintLiteral<V, U>,
        predicate: Predicate<AP, FP>,
        rhs: ConstraintLiteral<V, U>,
    ) -> Result<Self, InvalidConstraint> {
        if matches!(predicate, Predicate::Assign(_)) && matches!(rhs, ConstraintLiteral::Value(_)) {
            return Err(InvalidConstraint::AssignToValue);
        }
        Ok(Self {
            lhs,
            predicate,
            rhs,
        })
    }

    /// Return all variable assignments that would satisfy the constraint.
    ///
    /// This will panic if an `AssignPredicate` results in an invalid variable
    /// binding, or if the constraint is malformed.
    pub fn satisfy<S, D>(&self, data: &D, scope: S) -> Result<Vec<S>, LiteralEvalError>
    where
        S: Clone + VariableScope<V, U>,
        AP: AssignPredicate<D = D>,
        FP: FilterPredicate<D = D>,
    {
        let subject = self.lhs.evaluate(&scope)?;
        match &self.predicate {
            Predicate::Assign(ap) => {
                let objects = ap.check_assign(data, subject);
                let ConstraintLiteral::Variable(rhs) = &self.rhs else {
                    panic!("Invalid constraint: rhs of AssignPredicate is a value");
                };
                Ok(objects
                    .into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        scope.bind(rhs, obj).unwrap();
                        scope
                    })
                    .collect())
            }
            Predicate::Filter(fp) => {
                let object = self.rhs.evaluate(&scope)?;
                if fp.check(data, subject, object) {
                    Ok([scope].to_vec())
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }
}
