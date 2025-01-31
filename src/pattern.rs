//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use crate::{constraint_class::ConstraintClass, indexing::IndexKey};

/// A rank value used to prioritize branch classes.
///
/// Lower is better.
pub type ClassRank = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represents whether a pattern or constraint is satisfiable
pub enum Satisfiable<P = ()> {
    /// The pattern or constraint is satisfiable
    Yes(P),
    /// The pattern or constraint is not satisfiable
    No,
    /// The pattern or constraint is a tautology (i.e. is always satisifed).
    Tautology,
}

#[derive(Debug, Clone)]
/// Represents a selection of predicates for pattern matching
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
    /// The type for partially satisfied patterns.
    type PartialPattern: PartialPattern<Constraint = Self::Constraint, Key = Self::Key>;
    /// The constraint type used to express the pattern.
    type Constraint;

    /// List of required bindings to match the pattern.
    fn required_bindings(&self) -> Vec<Self::Key>;

    /// Extract a partial pattern for further processing.
    fn into_partial_pattern(self) -> Self::PartialPattern;
}

/// Partially satisfied patterns.
///
/// Provide the logic for constructing and simplifying patterns as constraints
/// get applied to it.
pub trait PartialPattern: Ord + Clone {
    /// The type of predicates used in the pattern.
    type Constraint: Ord + Clone;
    /// A partition of all predicates into mutually exclusive sets.
    type ConstraintClass: ConstraintClass<Self::Constraint>;
    /// The constraint key type.
    type Key: IndexKey;

    /// Get all constraints that are useful for pattern matching `self`.
    fn nominate(&self) -> impl Iterator<Item = Self::Constraint> + '_;

    /// Simplify patterns conditioned on which transitions are taken.
    ///
    /// All transitions belong to the `cls` class.
    ///
    /// For each transition in `transitions`, return whether the pattern is
    /// still satisfiable and, if so, the pattern equivalent to `self` when
    /// conditioned on that transition.
    ///
    /// If the returned vector is empty, the pattern will skip this round of
    /// constraint evaluation and be added to the "fail" transition instead.
    ///
    /// The input `transitions` is never empty. The return value should have
    /// the same length as `transitions`, or be empty.
    fn apply_transitions(
        &self,
        transitions: &[Self::Constraint],
        cls: &Self::ConstraintClass,
    ) -> Vec<Satisfiable<Self>>;

    /// Check whether the partial pattern is satisfiable.
    fn is_satisfiable(&self) -> Satisfiable;
}
