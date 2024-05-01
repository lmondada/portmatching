// TODO: reactivate this
// #![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use std::fmt::Debug;

pub mod automaton;
pub mod matcher;
pub mod new_api_v04;
mod pattern;
// #[cfg(feature = "portgraph")]
// pub mod portgraph;
pub mod weighted_graph;

pub mod utils;

pub use matcher::{ManyMatcher, NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use weighted_graph::WeightedGraphRef;

use rustc_hash::{FxHashMap, FxHashSet};

use std::hash::Hash;

/// A type that variable resolve to when satisfying constraints.
///
/// This is automatically implemented for any value that implements
/// `Copy`, `Eq`, `Hash`, and `Ord`.
pub trait Universe: Clone + Eq + Hash + Debug {}

impl<U: Clone + Eq + Hash + Debug> Universe for U {}

pub trait EdgeProperty: Clone + Ord + Hash {
    type OffsetID: Eq + Copy;

    fn reverse(&self) -> Option<Self>;

    fn offset_id(&self) -> Self::OffsetID;
}

pub trait NodeProperty: Clone + Hash + Ord {}

impl<U: Clone + Hash + Ord> NodeProperty for U {}

impl<A: Copy + Ord + Hash> EdgeProperty for (A, A) {
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

    fn offset_id(&self) -> Self::OffsetID {}
}

pub type HashMap<K, V> = FxHashMap<K, V>;
pub type HashSet<T> = FxHashSet<T>;
