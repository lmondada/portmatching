//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use crate::{constraint::ConstraintTag, indexing::IndexKey, Constraint, ConstraintEvaluator, Tag};

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

/// A pattern for pattern matching.
pub trait Pattern {
    /// The type for partially satisfied patterns.
    type PartialPattern: PartialPattern<
        Predicate = Self::Predicate,
        Key = Self::Key,
        Evaluator = Self::Evaluator,
        Tag = Self::Tag,
    >;
    /// Error for conversion to partial pattern failure
    type Error: std::fmt::Debug;

    // These four types are just aliases that can be derived from the above
    // types but simplify user code.

    /// The type of predicates used in the pattern.
    type Predicate: ConstraintTag<Self::Key, Tag = Self::Tag>;
    /// The type of keys used in the pattern.
    type Key: IndexKey;
    /// The type of tags for the pattern.
    type Tag: Tag<Self::Key, Self::Predicate, Evaluator = Self::Evaluator>;
    /// The type of evaluator for the pattern.
    type Evaluator: ConstraintEvaluator<Key = Self::Key>;

    /// List of required bindings to match the pattern.
    fn required_bindings(&self) -> Vec<Self::Key>;

    /// Extract a partial pattern for further processing.
    fn try_into_partial_pattern(self) -> Result<Self::PartialPattern, Self::Error>;
}

/// Partially satisfied patterns.
///
/// Provide the logic for constructing and simplifying patterns as constraints
/// get applied to it.
pub trait PartialPattern: Ord + Clone {
    /// The type of predicates used in the pattern.
    type Predicate: Ord + Clone + ConstraintTag<Self::Key, Tag = Self::Tag>;
    /// The constraint key type.
    type Key: IndexKey;

    // These two types are just aliases that can be derived from the above
    // types but simplify user code.

    /// The type of tags for the pattern.
    type Tag: Tag<Self::Key, Self::Predicate, Evaluator = Self::Evaluator>;
    /// The type of evaluator for the pattern.
    type Evaluator: ConstraintEvaluator<Key = Self::Key>;

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
        transitions: &[Constraint<Self::Key, Self::Predicate>],
        cls: &Self::Tag,
    ) -> Vec<Satisfiable<Self>>;

    /// Check whether the partial pattern is satisfiable.
    fn is_satisfiable(&self) -> Satisfiable;
}
