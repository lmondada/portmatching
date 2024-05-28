//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use super::{
    predicate::{ArityPredicate, AssignPredicate, FilterPredicate, Predicate},
    variable::VariableScope,
};
use itertools::Itertools;
use std::{cmp, fmt::Debug};
use thiserror::Error;

/// A variable `V` or value `U` argument in a constraint.
///
/// Literals are either a value from the predicate universe, or a variable
/// that will be bound at runtime to a value in the universe.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
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
/// Given by a predicate of arity N and a vector of N arguments. For assign
/// predicates, N >= 1 and the first argument is always a variable: the variable
/// to be bound by the assignment.
///
///
/// ## Total ordering of constraints
/// Constraints implement Ord, determining the order in which constraints
/// will be satisfied during pattern matching. An assignment of variable `v`
/// will be smaller than any filter predicate containing `v` or any variable
/// larger than `v`.
///
/// For any totally ordered vector of constraints, the following must hold:
///  - For AssignPredicates, all but the first argument must be constants or be
///    variables bound by previous constraints.
///  - For FilterPredicates, all arguments must be bound by previous constraints
///    if they are variables.
///
#[derive(Clone, PartialEq, Eq)]
pub struct Constraint<V, U, AP, FP> {
    predicate: Predicate<AP, FP>,
    args: Vec<ConstraintLiteral<V, U>>,
}

impl<V, U, AP, FP> Constraint<V, U, AP, FP>
where
    V: Debug,
    AP: ArityPredicate,
    FP: ArityPredicate,
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

    /// The constraint predicate
    pub fn predicate(&self) -> &Predicate<AP, FP> {
        &self.predicate
    }

    /// Return the arity of the constraint.
    pub fn arity(&self) -> usize {
        let arity = self.predicate.arity();
        assert_eq!(self.args.len(), arity, "invalid constraint: arity mismatch");
        arity
    }

    /// Return all variable assignments that would satisfy the constraint.
    ///
    /// This will panic if an `AssignPredicate` results in an invalid variable
    /// binding, or if the constraint is malformed.
    pub fn satisfy<S, D>(&self, data: &D, scope: S) -> Result<Vec<S>, InvalidConstraint>
    where
        S: Clone + VariableScope<V, U>,
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
    {
        let args = |n_args| {
            let start_ind = self.args.len() - n_args;
            self.args[start_ind..]
                .iter()
                .map(|arg| arg.evaluate(&scope))
                .collect::<Result<Vec<_>, _>>()
        };
        match &self.predicate {
            Predicate::Assign(ap) => {
                let rhs = ap.check_assign(data, &args(self.arity() - 1)?);
                let Some(ConstraintLiteral::Variable(lhs)) = &self.args.first() else {
                    panic!("Invalid constraint: rhs of AssignPredicate is not a variable");
                };
                Ok(rhs
                    .into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        scope.bind(lhs, obj).unwrap();
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

    /// Return the variable that is assigned by this constraint, if any.
    ///
    /// Returns None if the constraint is not an AssignPredicate.
    pub fn assigned_variable(&self) -> Option<&V> {
        match &self {
            &Constraint {
                predicate: Predicate::Assign(_),
                args,
            } => {
                let Some(ConstraintLiteral::Variable(var)) = args.first() else {
                    panic!("Invalid constraint: rhs of AssignPredicate is not a variable");
                };
                Some(var)
            }
            _ => None,
        }
    }
}

impl<V, U, AP, FP> PartialOrd for Constraint<V, U, AP, FP>
where
    V: Ord,
    U: Ord,
    AP: PartialOrd,
    FP: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.cmp_key().partial_cmp(&other.cmp_key())
    }
}

type CmpKey<'a, V, U, AP, FP> = (
    Option<&'a ConstraintLiteral<V, U>>,
    &'a Predicate<AP, FP>,
    &'a [ConstraintLiteral<V, U>],
);
impl<V: Ord, U: Ord, AP, FP> Constraint<V, U, AP, FP> {
    fn cmp_key(&self) -> CmpKey<V, U, AP, FP> {
        match &self.predicate {
            Predicate::Assign(_) => self.assign_key(),
            Predicate::Filter(_) => self.filter_key(),
        }
    }
    fn assign_key(&self) -> CmpKey<V, U, AP, FP> {
        (self.args.first(), &self.predicate, &self.args)
    }

    fn filter_key(&self) -> CmpKey<V, U, AP, FP> {
        (self.args.iter().max(), &self.predicate, &self.args)
    }
}

impl<V, U, AP, FP> Ord for Constraint<V, U, AP, FP>
where
    V: Ord,
    U: Ord,
    AP: Ord,
    FP: Ord,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<V, U, AP, FP> Debug for Constraint<V, U, AP, FP>
where
    Predicate<AP, FP>: Debug,
    ConstraintLiteral<V, U>: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}]", self.predicate)?;
        let args_str = self
            .args
            .iter()
            .map(|arg| format!("{:?}", arg))
            .collect_vec();
        write!(f, "({:?})", args_str.join(", "))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct APredicate;
    impl ArityPredicate for APredicate {
        fn arity(&self) -> usize {
            1
        }
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct FPredicate;
    impl ArityPredicate for FPredicate {
        fn arity(&self) -> usize {
            2
        }
    }

    fn ass_pred(a: ConstraintLiteral<i32, i32>) -> Constraint<i32, i32, APredicate, FPredicate> {
        Constraint::try_new(Predicate::Assign(APredicate), vec![a]).unwrap()
    }

    fn filt_pred(
        a: ConstraintLiteral<i32, i32>,
        b: ConstraintLiteral<i32, i32>,
    ) -> Constraint<i32, i32, APredicate, FPredicate> {
        Constraint::try_new(Predicate::Filter(FPredicate), vec![a, b]).unwrap()
    }

    #[test]
    fn test_constraint_ordering() {
        let a = ass_pred(ConstraintLiteral::<i32, i32>::Variable(1));
        let b = filt_pred(
            ConstraintLiteral::<i32, i32>::Variable(1),
            ConstraintLiteral::<i32, i32>::Variable(0),
        );
        // For same literal, an AssignPredicate should be smaller
        assert!(a < b);

        let a = ass_pred(ConstraintLiteral::<i32, i32>::Variable(3));
        let b = filt_pred(
            ConstraintLiteral::<i32, i32>::Variable(4),
            ConstraintLiteral::<i32, i32>::Variable(0),
        );
        // For smaller literal in assignment, AssignPredicate should still be smaller
        assert!(a < b);

        let a = ass_pred(ConstraintLiteral::<i32, i32>::Variable(3));
        let b = filt_pred(
            ConstraintLiteral::<i32, i32>::Variable(1),
            ConstraintLiteral::<i32, i32>::Variable(0),
        );
        // For larger literal in assignment, AssignPredicate should be larger
        assert!(a > b);
    }
}
