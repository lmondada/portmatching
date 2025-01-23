//! Traits to define predicates.

use std::{borrow::Borrow, collections::BTreeSet};

use crate::{
    constraint::InvalidConstraint,
    indexing::{Binding, IndexKey},
    pattern::Satisfiable,
    BindMap, Constraint, IndexedData,
};

/// Define the arity of a predicate
///
/// Predicates always operate on a fixed number of arguments, as given by
/// [`ArityPredicate::arity`].
pub trait ArityPredicate: Clone + Ord {
    /// The number of arguments this predicate expects
    fn arity(&self) -> usize;

    /// Convert a predicate with keys into a constraint
    ///
    /// Fails if the predicate arity does not match the number of keys
    fn try_into_constraint<K>(
        self,
        keys: Vec<K>,
    ) -> Result<Constraint<K, Self>, InvalidConstraint> {
        Constraint::try_new(self, keys)
    }
}

/// Evaluate predicates against data and bindings
///
/// Define how predicates are evaluated against concrete data and bindings
/// to produce boolean results.
pub trait EvaluatePredicate<Data, Value>: ArityPredicate {
    /// Check if this predicate holds for the given bindings and data
    ///
    /// # Arguments
    /// * `bindings` - The bound values to check against
    /// * `data` - The data context for evaluation
    fn check(&self, bindings: &[impl Borrow<Value>], data: &Data) -> bool;
}

/// Implement on a predicate to define its conditional logic within
/// [`super::PartialConstraintPattern`]s.
///
/// This trait defines how constraints simplify when conditioned on other constraints.
pub trait ConditionalPredicate<K>: Clone + Ord + Sized {
    /// Compute equivalent constraint when conditioned on an other constraint
    /// of the same class.
    ///
    /// `prev_constraints` is the set of constraints that have been evaluated
    /// so far (useful for a deterministic branch selector, in which case this
    /// means they were not satisfied).
    fn condition_on(
        &self,
        keys: &[K],
        known_constraints: &BTreeSet<Constraint<K, Self>>,
        prev_constraints: &[Constraint<K, Self>],
    ) -> Satisfiable<Constraint<K, Self>>;
}

/// Implement on a predicate to define the constraint classes that constraints
/// belongs to
pub trait GetConstraintClass<K> {
    /// Sets of constraints that can be evaluated together form branch classes.
    type ConstraintClass: Ord;

    /// All classes the constraint made of `self` and `keys` belongs to
    fn get_classes(&self, keys: &[K]) -> Vec<Self::ConstraintClass>;
}

impl<K, P> Constraint<K, P> {
    /// Evaluate the constraint given the subject data and index map.
    ///
    /// # Arguments
    ///
    /// * `data` - The data against which the constraint is evaluated
    /// * `known_bindings` - The current index map containing key-value bindings
    ///
    /// # Returns
    ///
    /// `Result<bool, InvalidConstraint>` - Ok(true) if the constraint is satisfied,
    /// Ok(false) if it's not, or an Err if there's an invalid constraint.
    pub fn is_satisfied<D: IndexedData<K>>(
        &self,
        data: &D,
        known_bindings: &D::BindMap,
    ) -> Result<bool, InvalidConstraint>
    where
        P: EvaluatePredicate<D, D::Value>,
        K: IndexKey,
    {
        let args = self
            .required_bindings()
            .iter()
            .map(|key| {
                let Binding::Bound(value) = known_bindings.get_binding(key) else {
                    return Err(InvalidConstraint::UnboundVariable(format!("{key:?}")));
                };
                Ok(value.borrow().clone())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self.predicate().check(&args, data))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Borrow;

    use itertools::Itertools;
    use rstest::rstest;

    use crate::constraint::{ConstraintPattern, EvaluatePredicate, PartialConstraintPattern};
    use crate::indexing::tests::TestData;

    use super::ArityPredicate;

    pub type TestKey = &'static str;
    pub type TestPattern = ConstraintPattern<TestKey, TestPredicate>;
    pub type TestPartialPattern = PartialConstraintPattern<TestKey, TestPredicate>;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) enum TestPredicate {
        // BranchClass One
        AreEqualOne,
        NotEqualOne,
        // BranchClass Two
        AreEqualTwo,
        AlwaysTrueTwo,
        // BranchClass Three
        NeverTrueThree,  // take one arg
        AlwaysTrueThree, // take one arg
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub(crate) enum TestBranchClass {
        One(TestKey, TestKey),
        Two(TestKey, TestKey),
        Three,
    }

    impl ArityPredicate for TestPredicate {
        fn arity(&self) -> usize {
            use TestPredicate::*;

            match self {
                AlwaysTrueThree | NeverTrueThree => 1,
                AreEqualOne | NotEqualOne | AreEqualTwo | AlwaysTrueTwo => 2,
            }
        }
    }

    impl EvaluatePredicate<TestData, usize> for TestPredicate {
        fn check(&self, bindings: &[impl Borrow<usize>], TestData: &TestData) -> bool {
            use TestPredicate::*;

            let args = bindings.iter().collect_tuple();
            match self {
                AreEqualOne | AreEqualTwo => {
                    let (a, b) = args.unwrap();
                    a.borrow() == b.borrow()
                }
                NotEqualOne => {
                    let (a, b) = args.unwrap();
                    a.borrow() != b.borrow()
                }
                AlwaysTrueThree | AlwaysTrueTwo => true,
                NeverTrueThree => false,
            }
        }
    }

    #[rstest]
    #[case(TestPredicate::AreEqualOne, vec![2, 2])]
    fn test_arity_match(#[case] predicate: TestPredicate, #[case] args: Vec<usize>) {
        assert!(predicate.check(&args, &TestData));
    }

    // TODO: more tests
}
