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

#[cfg(test)]
pub(crate) mod tests {
    use crate::predicate::{
        tests::{TestKey, TestPredicate},
        PredicatePatternDefaultSelector,
    };

    pub type TestBranchSelector = PredicatePatternDefaultSelector<TestKey, TestPredicate>;
}
