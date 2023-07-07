#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod automaton;
mod graph_traits;
pub mod matcher;
pub mod patterns;
pub(crate) mod predicate;
// mod symbol_map;

pub mod utils;

pub use graph_traits::GraphNodes;
pub use matcher::{ManyMatcher, NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use patterns::{Pattern, UnweightedPattern, WeightedPattern};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
// use symbol_map::SymbolMap;

use std::hash::{BuildHasherDefault, Hash};

pub trait Universe: Copy + Eq + Hash + Ord {}

impl<U: Copy + Eq + Hash + Ord> Universe for U {}

pub trait EdgeProperty: Copy + Ord + Hash {
    fn reverse(&self) -> Option<Self>;
}

pub trait NodeProperty: Copy + Hash + Ord {}

impl<U: Copy + Hash + Ord> NodeProperty for U {}

impl<A: Copy + Ord + Hash> EdgeProperty for (A, A) {
    fn reverse(&self) -> Option<Self> {
        (self.1, self.0).into()
    }
}

type BiMap<S, U> =
    bimap::BiHashMap<S, U, BuildHasherDefault<FxHasher>, BuildHasherDefault<FxHasher>>;
type HashSet<S> = FxHashSet<S>;
type HashMap<K, V> = FxHashMap<K, V>;
