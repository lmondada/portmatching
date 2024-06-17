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

use crate::HashSet;

/// A predicate with a fixed arity N.
pub trait ArityPredicate: Eq {
    /// Get Predicate arity
    fn arity(&self) -> usize;
}

/// A predicate for pattern matching.
///
/// Assign predicates must be of arity N >= 1 and bind the variable passed as its
/// first argument.
///
/// It always holds `Assign < Filter`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Predicate<AP, FP> {
    Assign(AP),
    Filter(FP),
}

impl<AP: ArityPredicate, FP: ArityPredicate> ArityPredicate for Predicate<AP, FP> {
    fn arity(&self) -> usize {
        match self {
            Predicate::Assign(ap) => ap.arity(),
            Predicate::Filter(fp) => fp.arity(),
        }
    }
}

/// A N-ary predicate that can be queried to return valid LHS bindings.
///
/// A predicate of the form `<var1> pred <var2> ... <varN>`. Given a binding
/// for <var2> to <varN>, the predicate returns a set of valid values for
/// <var1> for the given input data.
///
/// The arity N of an assign predicate must be N >= 1.
///
/// ## Parameter types
/// - `U`: The universe of valid symbols in the problem domain.
/// - `D`: The input data type in the problem domain.
pub trait AssignPredicate<U, D>: ArityPredicate {
    /// Find set of variable assignments that satisfy the predicate.
    ///
    /// `values` must be of length `arity() - 1` and correspond to the values
    /// of the last `arity() - 1` arguments in the predicate.
    fn check_assign(&self, data: &D, args: &[&U]) -> HashSet<U>;
}

/// A N-ary predicate for a given data and variable bindings.
///
/// A predicate of the form `<var1> pred <var2> ... <varN>`. Given bindings for
/// <var1> to <varN>, the predicate checks if it's satisfied on those values.
///
/// ## Parameter types
/// - `U`: The universe of valid symbols in the problem domain.
/// - `D`: The input data type in the problem domain.
pub trait FilterPredicate<U, D>: ArityPredicate {
    /// Check if the predicate is satisfied by the given data and values.
    ///
    /// `values` must be of length `arity()`.
    fn check(&self, data: &D, args: &[&U]) -> bool;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{predicate::Predicate, ArityPredicate, AssignPredicate, FilterPredicate, HashSet};

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct AssignEq;
    impl ArityPredicate for AssignEq {
        fn arity(&self) -> usize {
            2
        }
    }
    impl AssignPredicate<usize, ()> for AssignEq {
        fn check_assign(&self, _: &(), args: &[&usize]) -> HashSet<usize> {
            assert_eq!(args.len(), 1);
            HashSet::from_iter(args.iter().cloned().cloned())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct FilterEq;
    impl ArityPredicate for FilterEq {
        fn arity(&self) -> usize {
            2
        }
    }
    impl FilterPredicate<usize, ()> for FilterEq {
        fn check(&self, _: &(), args: &[&usize]) -> bool {
            let [arg0, arg1] = args else {
                panic!("Invalid constraint: arity mismatch");
            };
            arg0 == arg1
        }
    }

    type TestPredicate = Predicate<AssignEq, FilterEq>;

    #[test]
    fn test_arity_filter() {
        let p = TestPredicate::Filter(FilterEq);
        assert_eq!(p.arity(), 2);
    }

    #[test]
    fn test_arity_assign() {
        let p = TestPredicate::Assign(AssignEq);
        assert_eq!(p.arity(), 2);
    }
}
