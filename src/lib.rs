// TODO: reactivate this
// #![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod automaton;
pub mod matcher;
pub mod patterns;
pub(crate) mod predicate;
pub mod weighted_graph;

pub mod utils;

pub use matcher::{ManyMatcher, NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use patterns::{Pattern, UnweightedPattern, WeightedPattern};
pub use weighted_graph::WeightedGraphRef;

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use std::hash::{BuildHasherDefault, Hash};

pub trait Universe: Copy + Eq + Hash + Ord {}

impl<U: Copy + Eq + Hash + Ord> Universe for U {}

pub trait EdgeProperty: Clone + Ord + Hash {
    type OffsetID: Eq + Copy;

    fn reverse(&self) -> Option<Self>;

    /// Properties with the same ID are mutually exclusive (or identical).
    ///
    /// If None, the property is never exclusive.
    fn property_id(&self) -> Option<Self::OffsetID>;
}

pub trait NodeProperty: Clone + Hash + Ord {}

impl<U: Clone + Hash + Ord> NodeProperty for U {}

impl<A: Copy + Ord + Hash> EdgeProperty for (A, A) {
    type OffsetID = A;

    fn reverse(&self) -> Option<Self> {
        (self.1, self.0).into()
    }

    fn property_id(&self) -> Option<Self::OffsetID> {
        Some(self.0)
    }
}

impl EdgeProperty for () {
    type OffsetID = ();

    fn reverse(&self) -> Option<Self> {
        ().into()
    }

    fn property_id(&self) -> Option<Self::OffsetID> {
        Some(())
    }
}

type BiMap<S, U> =
    bimap::BiHashMap<S, U, BuildHasherDefault<FxHasher>, BuildHasherDefault<FxHasher>>;
type HashSet<S> = FxHashSet<S>;
pub type HashMap<K, V> = FxHashMap<K, V>;
