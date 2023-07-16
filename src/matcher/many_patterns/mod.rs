//! Pattern matchers for many patterns simultaneously.
//!
//! The [`NaiveManyMatcher`] is a simple matcher that uses [`super::SinglePatternMatcher`]s
//! to match each pattern separately.
//!
//! The [`LineGraphTrie`] is a more sophisticated matcher that uses a graph trie
//! data structure to match all patterns at once.
mod automaton;
mod naive;

use derive_more::{From, Into};
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[doc(inline)]
pub use automaton::{ManyMatcher, UnweightedManyMatcher};
#[doc(inline)]
pub use naive::NaiveManyMatcher;

/// ID for a pattern in a [`ManyPatternMatcher`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash)]
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
