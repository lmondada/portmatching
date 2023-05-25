#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod constraint;
pub mod graph_tries;
pub mod matcher;
pub mod pattern;
pub mod utils;

pub use constraint::{Constraint, Skeleton};

pub use matcher::{
    ManyPatternMatcher, Matcher, NaiveManyMatcher, PatternID, SinglePatternMatcher,
    TrieConstruction, TrieMatcher,
};
pub use pattern::{Pattern, UnweightedPattern, WeightedPattern};
