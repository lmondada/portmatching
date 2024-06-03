#![doc = include_str!("../README.md")]

use std::fmt::Debug;

pub mod automaton;
pub mod constraint;
pub mod matcher;
pub mod mutex_tree;
pub mod pattern;
pub mod predicate;
pub mod utils;
pub mod variable;
// #[cfg(feature = "portgraph")]
// pub mod portgraph;

pub use constraint::{Constraint, ConstraintLiteral, ConstraintType};
pub use matcher::{ManyMatcher, NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use pattern::Pattern;
pub use predicate::{ArityPredicate, AssignPredicate, FilterPredicate};
pub use variable::{VariableNaming, VariableScope};

use rustc_hash::{FxHashMap, FxHashSet};

use std::hash::Hash;

/// A type that variable resolve to when satisfying constraints.
///
/// This is automatically implemented for any value that implements
/// `Copy`, `Eq`, `Hash`, and `Ord`.
pub trait Universe: Clone + Eq + Hash + Debug {}

/// A host data type that can be iterated over for root candidates.
///
/// Provides an iterator over root candidates in the universe `U`.
pub trait IterRootCandidates {
    /// The type of the root candidates.
    type U;

    /// Iterate over the root candidates.
    fn root_candidates(&self) -> impl Iterator<Item = Self::U>;
}

impl<U: Clone + Eq + Hash + Debug> Universe for U {}

pub type HashMap<K, V> = FxHashMap<K, V>;
pub type HashSet<T> = FxHashSet<T>;
