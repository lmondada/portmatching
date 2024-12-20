//! Patterns for pattern matching.
//!
//! This module provides the core abstractions for defining patterns that can be
//! used in pattern matching operations. It includes:
//!
//! - The `Pattern` trait, which defines the basic interface for all patterns.
//! - The `ConcretePattern` trait, for patterns that can themselves be matched on.
//! - Error types related to pattern matching operations.

use std::collections::BTreeSet;

use crate::indexing::IndexKey;

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
    /// The predicate evaluatation logic.
    type Logic: PatternLogic<Constraint = Self::Constraint>;
    /// The type of constraints used in the pattern logic.
    type Constraint;

    /// List of required bindings to match the pattern.
    fn required_bindings(&self) -> Vec<Self::Key>;

    /// Extract the pattern logic for further processing.
    fn into_logic(self) -> Self::Logic;
}

/// The evaluation logic for a type of pattern.
pub trait PatternLogic: Ord + Clone {
    /// The type of predicates used in the pattern.
    type Constraint: Ord + Clone;
    /// A partition of all predicates into mutually exclusive sets.
    type BranchClass: Ord;

    /// The branch classes most pertinent to the pattern, along with a rank.
    ///
    /// The rank estimates the expected number of constraints in the class that
    /// are satisfied in both the pattern and random input data.
    ///
    /// i.e. if F is the set of constrains in a branch class and A c F is the
    /// subset of constraints satisfied by the pattern A(P), then the rank of
    /// the branch class F is E_G[ | { A in F | A(P) and A(G) } | ].
    ///
    /// The class with lowest overall rank will be selected.
    fn rank_classes(&self) -> impl Iterator<Item = (Self::BranchClass, ClassRank)>;

    /// Get all constraints in a class that are useful for pattern matching
    /// `self`.
    ///
    /// If the set is empty, then the pattern will be excluded from this
    /// class evaluation and be attached to the epsilon transition instead.
    fn nominate(&self, cls: &Self::BranchClass) -> BTreeSet<Self::Constraint>;

    /// Simplify patterns conditioned on which transitions are taken.
    ///
    /// For each transition in `transitions`, return whether the pattern is
    /// still satisfiable and, if so, the pattern equivalent to `self` when
    /// conditioned on that transition.
    fn apply_transitions(&self, transitions: &[Self::Constraint]) -> Vec<Satisfiable<Self>>;

    /// Check whether the pattern is satisfiable.
    fn is_satisfiable(&self) -> Satisfiable;
}
