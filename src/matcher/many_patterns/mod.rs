use std::{
    fmt,
};

use portgraph::{NodeIndex};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::pattern::{Pattern};

use super::Matcher;

mod address;
mod graph_trie;
mod line_based;
pub use line_based::LineGraphTrie;
// mod naive;
// pub use naive::{NaiveGraphTrie, NaiveManyPatternMatcher};

/// A match instance returned by a ManyPatternMatcher instance
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PatternMatch {
    pub id: PatternID,
    pub root: NodeIndex,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PatternID(pub usize);

impl fmt::Debug for PatternID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl fmt::Display for PatternID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ID({})", self.0)
    }
}

pub trait ManyPatternMatcher: Default + Matcher {
    type StateID;

    fn add_pattern(&mut self, pattern: Pattern) -> PatternID;

    fn from_patterns(patterns: Vec<Pattern>) -> Self {
        let mut obj = Self::default();
        for pattern in patterns {
            obj.add_pattern(pattern);
        }
        obj
    }
}

