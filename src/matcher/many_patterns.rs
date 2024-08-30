//! Pattern matchers for many patterns simultaneously.
//!
//! The [`NaiveManyMatcher`] is a simple matcher that uses [`super::SinglePatternMatcher`]s
//! to match each pattern separately.
//!
//! The [`LineGraphTrie`] is a more sophisticated matcher that uses a graph trie
//! data structure to match all patterns at once.
mod automaton;
mod naive;

use std::fmt;

#[doc(inline)]
pub use automaton::{ManyMatcher, PatternFallback};
#[doc(inline)]
pub use naive::NaiveManyMatcher;

use super::PatternID;

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
