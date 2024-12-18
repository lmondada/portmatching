//! # Branch Selector

use crate::indexing::IndexKey;

pub trait BranchSelector {
    /// The variable type
    type Key: IndexKey;

    /// The set of variables required to evaluate the branch predicates
    fn required_bindings(&self) -> &[Self::Key];
}

pub trait EvaluateBranchSelector<Data, Value>: BranchSelector {
    /// Evaluate the branch predicates and return the indices of the selected
    /// branches.
    ///
    /// The bindings should be given in order returned by `required_bindings`.
    /// If the variable could not be bound, pass `None`.
    fn eval(&self, bindings: &[Option<Value>], data: &Data) -> Vec<usize>;
}

pub trait CreateBranchSelector<C>: BranchSelector + Sized {
    fn create_branch_selector(constraints: Vec<C>) -> Self;
}

pub trait DisplayBranchSelector: BranchSelector {
    /// A string representation of the selector's branch class
    fn fmt_class(&self) -> String;

    /// A string representation of the n-th constraint
    fn fmt_nth_constraint(&self, n: usize) -> String;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::predicate::{
        tests::{TestBranchClass, TestKey, TestPredicate},
        PredicatePatternDefaultSelector,
    };

    pub type TestBranchSelector = PredicatePatternDefaultSelector<TestKey, TestPredicate>;

    impl super::DisplayBranchSelector for TestBranchSelector {
        fn fmt_class(&self) -> String {
            let Some(cls) = self.get_class() else {
                return String::new();
            };
            match cls {
                TestBranchClass::One(a, b) => format!("One({a}, {b})"),
                TestBranchClass::Two(a, b) => format!("Two({a}, {b})"),
                TestBranchClass::Three => "Three".to_string(),
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
