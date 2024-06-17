//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use super::{
    predicate::{ArityPredicate, AssignPredicate, FilterPredicate, Predicate},
    variable::{BindVariableError, VariableScope},
};
use itertools::Itertools;
use std::{cmp, fmt::Debug};
use thiserror::Error;

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
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Constraint<V, U, AP, FP> {
    predicate: Predicate<AP, FP>,
    args: Vec<ConstraintLiteral<V, U>>,
}

/// The type of constraint: either Assign or Filter.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintType<V> {
    /// Assignment constraint to variable V.
    Assign(V),
    /// Filter constraint.
    Filter,
}

/// A variable `V` or value `U` argument in a constraint.
///
/// Literals are either a value from the predicate universe, or a variable
/// that will be bound at runtime to a value in the universe.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintLiteral<V, U> {
    Variable(V),
    Value(U),
}

/// A heuristic whether a set of constraints should be turned into a deterministic
/// transition.
pub trait DetHeuristic
where
    Self: Sized,
{
    fn make_det(constraints: &[&Self]) -> bool;
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
    /// Construct a new value literal
    pub fn new_value(value: U) -> Self {
        ConstraintLiteral::Value(value)
    }

    /// Construct a new variable literal
    pub fn new_variable(variable: V) -> Self {
        ConstraintLiteral::Variable(variable)
    }

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

impl<V, U, AP, FP> Constraint<V, U, AP, FP> {
    /// The constraint predicate
    pub fn predicate(&self) -> &Predicate<AP, FP> {
        &self.predicate
    }
}

impl<V, U, AP, FP> Constraint<V, U, AP, FP>
where
    U: Debug,
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
            && matches!(args.first(), Some(ConstraintLiteral::Value(_)))
        {
            Err(InvalidConstraint::AssignToValue(format!(
                "{:?}",
                args.first().unwrap()
            )))
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
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
        V: Clone,
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
                let lhs = self
                    .assigned_variable()
                    .expect("Invalid constraint: No assigned variable in AssignPredicate");
                rhs.into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        scope.bind(lhs.clone(), obj)?;
                        Ok(scope)
                    })
                    .collect()
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

impl<V, U, AP, FP> Constraint<V, U, AP, FP> {
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

impl<'c, V: Clone, U, AP, FP> From<&'c Constraint<V, U, AP, FP>> for ConstraintType<V> {
    fn from(c: &'c Constraint<V, U, AP, FP>) -> Self {
        match c.assigned_variable() {
            Some(var) => ConstraintType::Assign(var.clone()),
            None => ConstraintType::Filter,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::predicate::tests::{AssignEq, FilterEq};
    use crate::HashMap;

    use super::*;
    pub(crate) type TestConstraint = Constraint<String, usize, AssignEq, FilterEq>;

    /// Construct a test assignment constraint
    pub(crate) fn assign_constraint(
        lhs: &str,
        rhs: ConstraintLiteral<String, usize>,
    ) -> TestConstraint {
        TestConstraint::try_binary_from_triple(
            ConstraintLiteral::Variable(lhs.to_string()),
            Predicate::Assign(AssignEq),
            rhs,
        )
        .unwrap()
    }

    /// Construct a test filter constraint
    pub(crate) fn filter_constraint(
        lhs: ConstraintLiteral<String, usize>,
        rhs: ConstraintLiteral<String, usize>,
    ) -> TestConstraint {
        TestConstraint::try_binary_from_triple(lhs, Predicate::Filter(FilterEq), rhs).unwrap()
    }

    #[test]
    fn test_construct_constraint() {
        let c = assign_constraint("x", ConstraintLiteral::Value(1));
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
        let c = assign_constraint("x", ConstraintLiteral::Value(1));
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("x"), Some(&1));
    }

    #[test]
    fn test_filter_satisfy_value_to_value() {
        let c = filter_constraint(ConstraintLiteral::Value(1), ConstraintLiteral::Value(1));
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_filter_satisfy_value_to_different_value() {
        let c = filter_constraint(ConstraintLiteral::Value(1), ConstraintLiteral::Value(2));
        let result = c.satisfy(&(), HashMap::default()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_assign_satisfy_value_to_variable() {
        let c = assign_constraint("y", ConstraintLiteral::Value(1));
        let scope = HashMap::from_iter([("x".to_string(), 1)]);
        let result = c.satisfy(&(), scope).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("y"), Some(&1));
    }

    #[test]
    fn test_constraint_ordering_same_literal() {
        let a = assign_constraint("b", ConstraintLiteral::Variable("b".to_string()));
        let b = filter_constraint(
            ConstraintLiteral::Variable("b".to_string()),
            ConstraintLiteral::Variable("a".to_string()),
        );
        // For same literal, an AssignPredicate should be smaller
        assert!(a < b);
    }

    #[test]
    fn test_constraint_ordering_smaller_literal() {
        let a = assign_constraint("d", ConstraintLiteral::Variable("d".to_string()));
        let b = filter_constraint(
            ConstraintLiteral::Variable("e".to_string()),
            ConstraintLiteral::Variable("a".to_string()),
        );
        // For smaller literal in assignment, AssignPredicate should still be smaller
        assert!(a < b);
    }

    #[test]
    fn test_constraint_ordering_larger_literal() {
        let a = assign_constraint("d", ConstraintLiteral::Variable("d".to_string()));
        let b = filter_constraint(
            ConstraintLiteral::Variable("b".to_string()),
            ConstraintLiteral::Variable("a".to_string()),
        );
        // For larger literal in assignment, AssignPredicate should be larger
        assert!(a > b);
    }

    #[test]
    fn test_assigned_variable() {
        let c = assign_constraint("x", ConstraintLiteral::Value(1));
        assert_eq!(c.assigned_variable(), Some(&"x".to_string()));
    }
}
