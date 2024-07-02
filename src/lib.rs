#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod automaton;
pub mod constraint;
pub mod indexing;
pub mod matcher;
pub mod mutex_tree;
pub mod pattern;
// #[cfg(feature = "portgraph")]
// pub mod portgraph;
pub mod predicate;
// pub mod string;
pub mod utils;

pub use constraint::Constraint;
pub use indexing::{IndexMap, IndexingScheme};
pub use matcher::{
    ManyMatcher, NaiveManyMatcher, PatternID, PatternMatch, PortMatcher, SinglePatternMatcher,
};
pub use pattern::Pattern;
pub use predicate::{ArityPredicate, Predicate};

use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) type HashMap<K, V> = FxHashMap<K, V>;
pub(crate) type HashSet<T> = FxHashSet<T>;
