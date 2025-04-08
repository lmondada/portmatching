//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use crate::{constraint::ConstraintTag, indexing::IndexKey, Constraint};

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
    /// All predicates in the tag may imply the pattern.
    All,
    /// A disjunction of predicates that may imply the pattern.
    Some(Vec<P>),
}

/// Type alias for the constraint type for a pattern.
pub type PatternConstraint<P> = Constraint<<P as Pattern>::Key, <P as Pattern>::Predicate>;

/// A pattern for pattern matching.
pub trait Pattern {
    /// The type for partially satisfied patterns.
    type PartialPattern: PartialPattern<Predicate = Self::Predicate, Key = Self::Key>;
    /// Error for conversion to partial pattern failure
    type Error: std::fmt::Debug;

    // These three types are just aliases that can be derived from the above
    // types but simplify user code.
    /// The type of predicates used in the pattern.
    type Predicate;
    /// The type of keys used in the pattern.
    type Key: IndexKey;

    /// List of required bindings to match the pattern.
    fn required_bindings(&self) -> Vec<Self::Key>;

    /// Extract a partial pattern for further processing.
    fn try_into_partial_pattern(self) -> Result<Self::PartialPattern, Self::Error>;
}

/// Type alias for the constraint type for a partial pattern.
pub type PartialPatternConstraint<P> =
    Constraint<<P as PartialPattern>::Key, <P as PartialPattern>::Predicate>;

/// Type alias for the tag type for a partial pattern.
pub type PartialPatternTag<P> =
    <<P as PartialPattern>::Predicate as ConstraintTag<<P as PartialPattern>::Key>>::Tag;

/// Partially satisfied patterns.
///
/// Provide the logic for constructing and simplifying patterns as constraints
/// get applied to it.
pub trait PartialPattern: Ord + Clone {
    /// The type of predicates used in the pattern.
    type Predicate: Ord + Clone + ConstraintTag<Self::Key>;
    /// The constraint key type.
    type Key: IndexKey;

    /// Get all constraints that are useful for pattern matching `self`.
    fn nominate(&self) -> impl Iterator<Item = Constraint<Self::Key, Self::Predicate>> + '_;

    /// Simplify patterns conditioned on which transitions are taken.
    ///
    /// All transitions belong to the `tag` tag.
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
        transitions: &[PartialPatternConstraint<Self>],
        cls: &PartialPatternTag<Self>,
    ) -> Vec<Satisfiable<Self>>;

    /// Check whether the partial pattern is satisfiable.
    fn is_satisfiable(&self) -> Satisfiable;
}
