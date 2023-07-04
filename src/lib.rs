#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod automaton;
pub mod matcher;
pub mod patterns;
pub(crate) mod predicate;

pub mod utils;

pub use matcher::{NaiveManyMatcher, PatternID, PortMatcher, SinglePatternMatcher};
pub use patterns::{Pattern, UnweightedPattern};

use std::hash::Hash;

pub trait Universe: Copy + Eq + Hash {}

impl<U: Copy + Eq + Hash> Universe for U {}

pub trait Property: Copy + Ord + Hash {}

impl<U: Copy + Ord + Hash> Property for U {}
