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

use std::collections::HashSet;
use std::hash::Hash;

/// A predicate for pattern matching.
#[derive(Clone, Debug)]
pub enum Predicate<AP, FP> {
    Assign(AP),
    Filter(FP),
}

/// A binary predicate that can be queried to return valid RHS bindings.
///
/// A predicate of the form `<var1> pred <var2>`. Given a binding for <var1>,
/// the predicate can return a set of valid values for <var2> for the given
/// input data.
pub trait AssignPredicate {
    /// The universe of valid symbols in the problem domain
    type U;
    /// The input data type in the problem domain.
    type D;

    /// Find set of variable assignments that satisfy the predicate.
    fn check_assign(&self, data: &Self::D, value: &Self::U) -> HashSet<Self::U>;
}

/// A binary predicate for a given data and variable bindings.
///
/// A predicate of the form `<var1> pred <var2>`. Given bindings for <var1>
/// and <var2> the predicate can check if it's satisfied on those values.
pub trait FilterPredicate {
    /// The universe of valid symbols in the problem domain.
    type U;
    /// The input data type in the problem domain.
    type D;

    /// Check if the predicate is satisfied by the given data and values.
    fn check(&self, data: &Self::D, value1: &Self::U, value2: &Self::U) -> bool;
}

impl<T> FilterPredicate for T
where
    T: AssignPredicate,
    <T as AssignPredicate>::U: Eq + Hash,
{
    type U = T::U;

    type D = T::D;

    fn check(&self, data: &Self::D, value1: &Self::U, value2: &Self::U) -> bool {
        self.check_assign(data, value1).contains(value2)
    }
}
