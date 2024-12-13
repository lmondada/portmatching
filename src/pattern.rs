//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.


use crate::indexing::IndexKey;

pub type ClassRank = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Satisfiable<P = ()> {
    /// The pattern is satisfiable.
    Yes(P),
    /// The pattern is not satisfiable.
    No,
    /// The pattern is a tautology (i.e. is always satisifed).
    Tautology,
}

pub enum PredicateSelection<P> {
    /// All predicates in the class may imply the pattern.
    All,
    /// A disjunction of predicates that may imply the pattern.
    Some(Vec<P>),
}

/// A pattern for pattern matching.
pub trait Pattern {
    /// The type of variable names used in the pattern.
    type Key: IndexKey;
    /// The predicate evaluatation logic.
    type Logic: PatternLogic<Constraint = Self::Constraint>;

    type Constraint;

    /// List of required bindings to match the pattern.
    fn required_bindings(&self) -> Vec<Self::Key>;

    /// Extract the pattern logic for further processing.
    fn into_logic(self) -> Self::Logic;
}

type ConditionalPattern<C, P> = (Option<C>, P);

/// The evaluation logic for a type of pattern.
pub trait PatternLogic: Ord + Clone {
    /// The type of predicates used in the pattern.
    type Constraint: Ord + Clone;
    /// A partition of all predicates into mutually exclusive sets.
    type BranchClass: Ord;

    /// The predicate classes used in the pattern, along with their rank.
    ///
    /// The rank indicates the probability of the pattern being retained when
    /// matching for the corresponding predicate class.
    fn get_branch_classes(&self) -> impl Iterator<Item = (Self::BranchClass, ClassRank)>;

    /// A pattern equivalent to `self` when conditioned on `constraints` and
    /// the branch class `cls`.
    ///
    /// This can be viewed as 'popping' a constraint from the pattern: return
    /// tuples where the first element is an optional constraint in `cls` and
    /// the second is the pattern equivalent to `self` when conditioned on that
    /// constraint and `constraints`. The iterator enumerates all possible
    /// decompositions of `self` into a constraint in `cls` and a pattern
    /// equivalent to `self`.
    fn condition_on<'p>(
        &self,
        cls: &Self::BranchClass,
        constraints: impl IntoIterator<Item = &'p Self::Constraint>,
    ) -> impl Iterator<Item = ConditionalPattern<Self::Constraint, Self>>
    where
        Self: 'p;

    /// Check whether the pattern is satisfiable.
    fn is_satisfiable(&self) -> Satisfiable;
}
