//! Predicates to define constraints and patterns.
//!
//! A predicate is a boolean-valued function, to be evaluated on some input
//! data and variable bindings.
//!
//! In theory, a binary predicate is thus a function D x U x U -> Bool.
//! However, to simplify using predicates in practice, we distinguish two types
//! of predicates:
//!
//! 1. [FilterPredicates] The "vanilla" case. Given data and variable bindings,
//!    evaluate the predicate and return a boolean value.
//! 2. [AssignPredicates] The "auto-complete" case. Given a binding for <var1>
//!    and input data, the predicate returns a set of values for <var2> that
//!    would satisfy the predicate.
//!
//! Note that formally, an `AssignPredicate` is just a `FilterPredicate` with a
//! more efficient way of resolving the constraint: it can always be viewed as a
//! `FilterPredicate` by calling `assign_check` and then checking that the binding
//! for <var2> is in the returned set.

use std::borrow::Borrow;

use crate::indexing::{DataValue, IndexedData};

/// A predicate with a fixed arity.
pub trait ArityPredicate: Eq + Clone {
    /// Get predicate arity
    fn arity(&self) -> usize;
}

/// A N-ary predicate evaluated on bindings and subject data.
///
/// A predicate of the form `pred <key1> ... <keyN>`. Given bindings for <key1>
/// to <keyN>, the predicate checks if it's satisfied on the values.
///
/// ## Parameter types
/// - `Data`: The subject data type on which predicates are evaluated.
pub trait Predicate<Data: IndexedData>: ArityPredicate {
    /// The error type for malformed predicates.
    type InvalidPredicateError: std::fmt::Display;

    /// Check if the predicate is satisfied by the given data and values.
    ///
    /// `values` must be of length [Predicate::arity].
    fn check(
        &self,
        data: &Data,
        args: &[impl Borrow<DataValue<Data>>],
    ) -> Result<bool, Self::InvalidPredicateError>;
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Borrow;
    use std::cmp;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::indexing::tests::TestData;
    use crate::indexing::DataValue;
    use crate::predicate::Predicate;

    use super::ArityPredicate;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub(crate) struct TestPredicate {
        pub(crate) arity: usize,
    }

    impl ArityPredicate for TestPredicate {
        fn arity(&self) -> usize {
            self.arity
        }
    }

    impl Predicate<TestData> for TestPredicate {
        type InvalidPredicateError = String;

        fn check(
            &self,
            _: &TestData,
            args: &[impl Borrow<DataValue<TestData>>],
        ) -> Result<bool, String> {
            if args.len() != self.arity {
                Err("Invalid constraint: arity mismatch".to_string())
            } else {
                Ok(args
                    .iter()
                    .tuple_windows()
                    .all(|(a, b)| a.borrow() == b.borrow()))
            }
        }
    }

    impl PartialOrd for TestPredicate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for TestPredicate {
        fn cmp(&self, other: &Self) -> cmp::Ordering {
            cmp::Reverse(self.arity).cmp(&cmp::Reverse(other.arity))
        }
    }

    #[rstest]
    #[case(2)]
    #[case(3)]
    fn test_arity_match(#[case] arity: usize) {
        let p = TestPredicate { arity };
        let args = vec![&3; p.arity()];
        assert!(p.check(&TestData, &args).unwrap());
    }

    #[test]
    fn test_arity_mismatch() {
        let p = TestPredicate { arity: 2 };
        let args = vec![&3];
        assert!(p.check(&TestData, &args).is_err());
    }
}
