//! A constraint is a triple (subj `pred obj).
//!
//! Subject and object may be variables or values.
//!
//! The predicate is either an AssignPredicate or a FilterPredicate.

use super::{
    predicate::{AssignPredicate, FilterPredicate, Predicate},
    variable::VariableScope,
};

/// A litteral for subject and object in constraints.
///
/// Litterals are either a value from the predicate universe, or a variable
/// that will be bound at runtime to a value in the universe.
pub enum ConstraintLitteral<V, U> {
    Variable(V),
    Value(U),
}

impl<V, U> ConstraintLitteral<V, U> {
    /// Evaluate a litteral to a value in U.
    ///
    /// If the litteral is a value, unwrap it. Otherwise, use the variable scope
    /// to resolve the binding.
    ///
    /// If the variable is not defined, this will panic.
    pub fn evaluate<'a>(&'a self, scope: &'a impl VariableScope<V, U>) -> &'a U {
        match &self {
            ConstraintLitteral::Variable(var) => scope.get(var).expect("Unbound variable"),
            ConstraintLitteral::Value(val) => val,
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
pub struct Constraint<V, U, AP, FP> {
    lhs: ConstraintLitteral<V, U>,
    predicate: Predicate<AP, FP>,
    rhs: ConstraintLitteral<V, U>,
}

/// Errors that occur when constructing constraints.
pub enum InvalidConstraint {
    /// Cannot assign a value if the RHS is not a variable
    AssignToValue,
}

impl<V, U, AP, FP> Constraint<V, U, AP, FP>
where
    AP: AssignPredicate<U = U>,
    FP: FilterPredicate<U = U>,
{
    /// Construct a constraint.
    ///
    /// Returns an error if the constraint is malformed, i.e. if the predicate is
    /// a Predicate::Assign and the object is not a variable.
    pub fn try_from_triple(
        lhs: ConstraintLitteral<V, U>,
        predicate: Predicate<AP, FP>,
        rhs: ConstraintLitteral<V, U>,
    ) -> Result<Self, InvalidConstraint> {
        if matches!(predicate, Predicate::Assign(_)) && matches!(rhs, ConstraintLitteral::Value(_))
        {
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
    pub fn satisfy<S, D>(&self, data: &D, scope: S) -> Vec<S>
    where
        S: Clone + VariableScope<V, U>,
        AP: AssignPredicate<D = D>,
        FP: FilterPredicate<D = D>,
    {
        let subject = self.lhs.evaluate(&scope);
        match &self.predicate {
            Predicate::Assign(ap) => {
                let objects = ap.check_assign(data, &subject);
                objects
                    .into_iter()
                    .map(|obj| {
                        let mut scope = scope.clone();
                        let ConstraintLitteral::Variable(rhs) = &self.rhs else {
                            panic!("Invalid constraint: rhs of AssignPredicate is a value");
                        };
                        scope.bind(rhs, obj).unwrap();
                        scope
                    })
                    .collect()
            }
            Predicate::Filter(fp) => {
                let object = self.rhs.evaluate(&scope);
                if fp.check(data, subject, object) {
                    [scope].to_vec()
                } else {
                    Vec::new()
                }
            }
        }
    }
}
