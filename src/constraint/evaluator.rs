//! Given a list of constraints that share a tag, a constraint evaluator
//! evaluates them efficiently and returns the indices of the satisfied
//! constraints.
//!
//! We provide default implementations for two types of evaluator
//!  - [`DefaultConstraintEvaluator`]: will evaluate all constraints one by one
//!    and return all the constraints that evaluated to true.
//!  - [`DeterministicConstraintEvaluator`]: will evaluate the constraints in
//!    order and returns the first constraint that evaluates to true
//!    (short-circuiting).

mod default;
mod det;
mod inner;

use inner::InnerEvaluator;

pub use default::DefaultConstraintEvaluator;
pub use det::DeterministicConstraintEvaluator;

use crate::indexing::IndexKey;

/// ConstraintEvaluators are used to evaluate all constraints in a state and
/// choose which transitions to descend into.
///
/// The required bindings of a constraint evaluator is the union of all bindings
/// required to evaluate its constraints.
pub trait ConstraintEvaluator {
    /// The variable type
    type Key: IndexKey;

    /// The set of variables required to evaluate the branch predicates.
    fn required_bindings(&self) -> &[Self::Key];

    /// A string representation of the n-th constraint in `self`.
    fn fmt_nth_constraint(&self, n: usize) -> String {
        format!("Constraint {}", n)
    }

    /// A short string summary of the constraint evaluator tag.
    fn summary(&self) -> String {
        format!("ConstraintEvaluator")
    }
}

/// Use a constraint evaluator to evaluate a set of constraints on values of type
/// `Values in `Data`.
pub trait EvaluateConstraints<Data, Value>: ConstraintEvaluator {
    /// Evaluate all constraints and return the indices of the satisfied constraints.
    ///
    /// The bindings should be given in order returned by `required_bindings`.
    /// If the variable could not be bound, pass `None`.
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        constraint::{
            tests::{TestKey, TestPredicate},
            DefaultConstraintEvaluator,
        },
        indexing::tests::TestData,
    };
    use delegate::delegate;

    use super::{ConstraintEvaluator, EvaluateConstraints};

    pub struct TestConstraintEvaluator(DefaultConstraintEvaluator<TestKey, TestPredicate>);

    impl TestConstraintEvaluator {
        pub fn from_constraints<'c>(
            constraints: impl IntoIterator<Item = (&'c TestPredicate, &'c [TestKey])>,
        ) -> Self {
            Self(DefaultConstraintEvaluator::from_constraints(constraints))
        }
    }

    impl ConstraintEvaluator for TestConstraintEvaluator {
        type Key = TestKey;

        fn required_bindings(&self) -> &[Self::Key] {
            self.0.required_bindings()
        }

        fn fmt_nth_constraint(&self, n: usize) -> String {
            format!("{:?}", self.0.predicates()[n])
        }

        fn summary(&self) -> String {
            format!("{:?}", self.0.get_tag().unwrap())
        }
    }

    impl EvaluateConstraints<TestData, usize> for TestConstraintEvaluator {
        delegate! {
            to self.0 {
                fn eval(&self, bindings: &[Option<usize>], data: &TestData) -> Vec<usize>;
            }
        }
    }
}
