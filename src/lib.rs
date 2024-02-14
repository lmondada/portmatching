// TODO: reactivate this
// #![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod automaton;
pub mod constraint;
pub mod matcher;
pub mod pattern;
// pub mod weighted_graph;

pub mod utils;

pub mod portgraph;

pub use matcher::{ManyMatcher, NaiveManyMatcher, PatternID, PatternMatch, SinglePatternMatcher};
pub use pattern::Pattern;
pub use constraint::{ScopeConstraint, Symbol};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use std::hash::BuildHasherDefault;

type BiMap<S, U> =
    bimap::BiHashMap<S, U, BuildHasherDefault<FxHasher>, BuildHasherDefault<FxHasher>>;
type HashSet<S> = FxHashSet<S>;
pub type HashMap<K, V> = FxHashMap<K, V>;
