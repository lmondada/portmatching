#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod addressing;
pub mod graph_tries;
pub mod matcher;
pub mod pattern;
pub mod utils;

pub use matcher::{
    BalancedTrieMatcher, DetTrieMatcher, ManyPatternMatcher, Matcher, NaiveManyMatcher,
    NonDetTrieMatcher, PatternID, SinglePatternMatcher,
};
pub use pattern::Pattern;
