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

impl<AP: AssignPredicate, FP: FilterPredicate> Predicate<AP, FP> {
    /// Get Predicate arity
    pub fn arity(&self) -> usize {
        match self {
            Predicate::Assign(ap) => ap.arity(),
            Predicate::Filter(fp) => fp.arity(),
        }
    }
}

/// A predicate with a fixed arity N.
pub trait ArityPredicate {
    /// Get Predicate arity
    fn arity(&self) -> usize;
}

/// A N-ary predicate that can be queried to return valid RHS bindings.
///
/// A predicate of the form `<var1>, <var2> ... pred <varN>`. Given a binding
/// for <var1> to <varN-1>, the predicate returns a set of valid values for
/// <varN> for the given input data.
///
/// The arity N of an assign predicate must be N >= 1.
pub trait AssignPredicate: ArityPredicate {
    /// The universe of valid symbols in the problem domain
    type U;
    /// The input data type in the problem domain.
    type D;

    /// Find set of variable assignments that satisfy the predicate.
    ///
    /// `values` must be of length `arity() - 1`.
    fn check_assign(&self, data: &Self::D, args: &[&Self::U]) -> HashSet<Self::U>;
}

/// A N-ary predicate for a given data and variable bindings.
///
/// A predicate of the form `<var1>, <var2> ... pred <varN>`. Given bindings for
/// <var1> to <varN>, the predicate checks if it's satisfied on those values.
pub trait FilterPredicate: ArityPredicate {
    /// The universe of valid symbols in the problem domain.
    type U;
    /// The input data type in the problem domain.
    type D;

    /// Check if the predicate is satisfied by the given data and values.
    ///
    /// `values` must be of length `arity()`.
    fn check(&self, data: &Self::D, args: &[&Self::U]) -> bool;
}

impl<T> FilterPredicate for T
where
    T: AssignPredicate,
    <T as AssignPredicate>::U: Eq + Hash,
{
    type U = T::U;

    type D = T::D;

    fn check(&self, data: &Self::D, args: &[&Self::U]) -> bool {
        assert!(self.arity() >= 1, "AssignPredicate arity must be >= 1");
        assert_eq!(args.len(), self.arity(), "invalid predicate data");

        self.check_assign(data, &args[..self.arity() - 1])
            .contains(args[self.arity() - 1])
    }
}
