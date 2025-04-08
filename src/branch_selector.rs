//! # Branch Selector

use crate::indexing::IndexKey;

/// BranchSelectors are used to evaluate all constraints in a state and choose
/// which transitions to descend into.
///
/// This base trait defines the interface useful both when constructing
/// branch selectors and when evaluating them.
pub trait BranchSelector {
    /// The variable type
    type Key: IndexKey;

    /// The set of variables required to evaluate the branch predicates
    fn required_bindings(&self) -> &[Self::Key];
}

/// Extend `BranchSelector` with the ability to evaluate data and produce values
/// of a specific type.
pub trait EvaluateBranchSelector<Data, Value>: BranchSelector {
    /// Evaluate the branch predicates and return the indices of the selected
    /// branches.
    ///
    /// The bindings should be given in order returned by `required_bindings`.
    /// If the variable could not be bound, pass `None`.
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
}

/// Extend `BranchSelector` with the ability to be constructed from
/// a collection of constraints.
pub trait CreateBranchSelector<C>: BranchSelector + Sized {
    /// Create a new branch selector from a vector of constraints
    ///
    /// # Arguments
    /// * `constraints` - A vector of constraints used to construct the selector
    fn create_branch_selector(constraints: Vec<C>) -> Self;
}

/// Extend `BranchSelector` with the ability to be formatted for display.
pub trait DisplayBranchSelector: BranchSelector {
    /// A string representation of the selector's branch tag
    fn fmt_tag(&self) -> String;

    /// A string representation of the n-th constraint
    fn fmt_nth_constraint(&self, n: usize) -> String;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::constraint::{
        tests::{TestConstraintTag, TestKey, TestPredicate},
        DefaultConstraintSelector,
    };

    pub type TestBranchSelector = DefaultConstraintSelector<TestKey, TestPredicate>;

    impl super::DisplayBranchSelector for TestBranchSelector {
        fn fmt_tag(&self) -> String {
            let Some(cls) = self.get_tag() else {
                return String::new();
            };
            match cls {
                TestConstraintTag::One(a, b) => format!("One({a}, {b})"),
                TestConstraintTag::Two(a, b) => format!("Two({a}, {b})"),
                TestConstraintTag::Three => "Three".to_string(),
            }
        }

        fn fmt_nth_constraint(&self, n: usize) -> String {
            match self.predicates()[n] {
                TestPredicate::AreEqualOne => "==".to_string(),
                TestPredicate::NotEqualOne => "!=".to_string(),
                TestPredicate::AreEqualTwo => "==".to_string(),
                TestPredicate::AlwaysTrueTwo => "TRUE".to_string(),
                TestPredicate::NeverTrueThree => "FALSE".to_string(),
                TestPredicate::AlwaysTrueThree => "TRUE".to_string(),
            }
        }
    }
}
