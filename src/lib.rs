#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod automaton;
mod graph_traits;
pub mod matcher;
pub mod patterns;
pub(crate) mod predicate;

pub mod utils;

pub use graph_traits::GraphNodes;
pub use matcher::{NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use patterns::{Pattern, UnweightedPattern};

use std::hash::Hash;

pub trait Universe: Copy + Eq + Hash {}

impl<U: Copy + Eq + Hash> Universe for U {}

pub trait EdgeProperty: Copy + Ord + Hash {
    fn reverse(&self) -> Option<Self>;
}

pub trait NodeProperty: Copy + Eq + Hash {}

impl<U: Copy + Eq + Hash> NodeProperty for U {}

impl<A: Copy + Ord + Hash> EdgeProperty for (A, A) {
    fn reverse(&self) -> Option<Self> {
        (self.1, self.0).into()
    }
}
