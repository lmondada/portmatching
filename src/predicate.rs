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

mod pattern;
pub use pattern::{PredicatePattern, PredicatePatternDefaultSelector};

use std::borrow::Borrow;

use crate::{constraint::InvalidConstraint, pattern::Satisfiable, Constraint};

pub trait ArityPredicate: Clone + Ord {
    fn arity(&self) -> usize;
}

/// A N-ary predicate evaluated on bindings and subject data.
///
/// A predicate of the form `pred <key1> ... <keyN>`. Given bindings for <key1>
/// to <keyN>, the predicate checks if it's satisfied on the values.
///
/// ## Parameter types
/// - `Data`: The subject data type on which predicates are evaluated.
pub trait Predicate<Data, Value>: ArityPredicate {
    fn check(&self, bindings: &[impl Borrow<Value>], data: &Data) -> bool;
}

pub trait ConstraintLogic<K>: Clone + Ord + Sized {
    type BranchClass: Ord;

    fn get_class(&self, keys: &[K]) -> Self::BranchClass;

    /// Compute equivalent constraint when conditioned on an other constraint
    /// of the same class.
    fn condition_on(
        &self,
        keys: &[K],
        condition: &Constraint<K, Self>,
    ) -> Satisfiable<Constraint<K, Self>>;

    fn try_into_constraint(self, keys: Vec<K>) -> Result<Constraint<K, Self>, InvalidConstraint>
    where
        Self: ArityPredicate,
    {
        Constraint::try_new(self, keys)
    }
}

impl<K, P> Constraint<K, P> {
    pub fn get_class(&self) -> P::BranchClass
    where
        P: ConstraintLogic<K>,
    {
        self.predicate().get_class(self.required_bindings())
    }

    pub fn condition_on(&self, condition: &Constraint<K, P>) -> Satisfiable<Constraint<K, P>>
    where
        P: ConstraintLogic<K>,
    {
        let keys = self.required_bindings();
        self.predicate().condition_on(keys, condition)
    }
}
#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Borrow;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::indexing::tests::TestData;
    use crate::pattern::Satisfiable;
    use crate::predicate::Predicate;
    use crate::Constraint;

    use super::{ArityPredicate, ConstraintLogic};

    pub type TestKey = &'static str;
    pub type TestPattern = super::PredicatePattern<TestKey, TestPredicate>;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) enum TestPredicate {
        AreEqual,
        NotEqual,
        PredTwo,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(crate) enum TestBranchClass {
        One(TestKey, TestKey),
        Two,
    }

    impl ArityPredicate for TestPredicate {
        fn arity(&self) -> usize {
            match self {
                TestPredicate::AreEqual => 2,
                TestPredicate::NotEqual => 2,
                TestPredicate::PredTwo => 0,
            }
        }
    }

    impl Predicate<TestData, usize> for TestPredicate {
        fn check(&self, bindings: &[impl Borrow<usize>], TestData: &TestData) -> bool {
            let (a, b) = bindings.into_iter().collect_tuple().unwrap();
            match self {
                TestPredicate::AreEqual => a.borrow() == b.borrow(),
                TestPredicate::NotEqual => a.borrow() != b.borrow(),
                TestPredicate::PredTwo => true,
            }
        }
    }

    impl ConstraintLogic<TestKey> for TestPredicate {
        type BranchClass = TestBranchClass;

        fn get_class(&self, keys: &[TestKey]) -> Self::BranchClass {
            let (a, b) = keys.into_iter().cloned().collect_tuple().unwrap();
            match self {
                Self::AreEqual => TestBranchClass::One(a, b),
                Self::NotEqual => TestBranchClass::One(a, b),
                Self::PredTwo => TestBranchClass::Two,
            }
        }

        fn condition_on(
            &self,
            keys: &[TestKey],
            condition: &Constraint<TestKey, Self>,
        ) -> Satisfiable<Constraint<TestKey, Self>> {
            assert_eq!(
                self.get_class(keys),
                condition.get_class(),
                "class mismatch in TestPredicate::condition_on"
            );
            assert_eq!(
                keys,
                condition.required_bindings(),
                "Cannot be in same class if keys are not the same"
            );

            if self == condition.predicate() {
                Satisfiable::Tautology
            } else {
                Satisfiable::No
            }
        }
    }

    #[rstest]
    #[case(TestPredicate::AreEqual, vec![2, 2])]
    fn test_arity_match(#[case] predicate: TestPredicate, #[case] args: Vec<usize>) {
        assert!(predicate.check(&args, &TestData));
    }

    // TODO: more tests
}
