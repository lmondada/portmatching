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

use std::{
    fmt::Debug,
    hash::{BuildHasherDefault, Hash},
};

pub trait Universe: Copy + Eq + Hash + Ord {}

impl<U: Copy + Eq + Hash + Ord> Universe for U {}

pub trait EdgeProperty: Copy + Ord + Hash + std::fmt::Debug {
    type OffsetID: Eq + Copy + Debug;

    fn reverse(&self) -> Option<Self>;

    fn offset_id(&self) -> Self::OffsetID;
}

pub trait NodeProperty: Copy + Hash + Ord + std::fmt::Debug {}

impl<U: Copy + Hash + Ord + std::fmt::Debug> NodeProperty for U {}

impl<A: Copy + Ord + Hash + Debug> EdgeProperty for (A, A) {
    type OffsetID = A;

    fn reverse(&self) -> Option<Self> {
        (self.1, self.0).into()
    }

    fn offset_id(&self) -> Self::OffsetID {
        self.0
    }
}

impl EdgeProperty for () {
    type OffsetID = ();

    fn reverse(&self) -> Option<Self> {
        ().into()
    }

    fn offset_id(&self) -> Self::OffsetID {
        
    }
}

type BiMap<S, U> =
    bimap::BiHashMap<S, U, BuildHasherDefault<FxHasher>, BuildHasherDefault<FxHasher>>;
type HashSet<S> = FxHashSet<S>;
type HashMap<K, V> = FxHashMap<K, V>;
