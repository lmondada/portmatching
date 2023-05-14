#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod addressing;
pub mod graph_tries;
pub mod matcher;
pub mod pattern;
pub mod utils;

pub use matcher::{
    BalancedTrieMatcher,
    ManyPatternMatcher,
    Matcher,
    NaiveManyMatcher,
    PatternID,
    SinglePatternMatcher,
    // DetTrieMatcher,
    // NonDetTrieMatcher,
};
pub use pattern::Pattern;
