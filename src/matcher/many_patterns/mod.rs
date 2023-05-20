//! Pattern matchers for many patterns simultaneously.
//!
//! The [`NaiveManyMatcher`] is a simple matcher that uses [`super::SinglePatternMatcher`]s
//! to match each pattern separately.
//!
//! The [`LineGraphTrie`] is a more sophisticated matcher that uses a graph trie
//! data structure to match all patterns at once.
mod naive;
mod trie_matcher;

use portgraph::NodeIndex;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Matcher;
use crate::pattern::Pattern;
#[doc(inline)]
pub use naive::NaiveManyMatcher;
#[doc(inline)]
pub use trie_matcher::{TrieConstruction, TrieMatcher};

/// A match instance returned by a ManyPatternMatcher instance.
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PatternMatch {
    /// The ID of the pattern that matches.
    pub id: PatternID,
    /// The root node of the match.
    ///
    /// The entire match can be recovered from the root mapping
    /// using [`crate::Pattern::get_boundary`].
    pub root: NodeIndex,
}

/// ID for a pattern in a [`ManyPatternMatcher`].
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

/// A trait for pattern matchers that can match many patterns at once.
///
/// This trait extends the [`Matcher`] trait with the ability to add
/// additional patterns to the matcher.
///
/// Note that not all matchers meant to match multiple patterns at once
/// implement this trait, as some matchers are obtained by converting
/// previously constructed matchers and thus do not support adding patterns
/// one-by-one.
pub trait ManyPatternMatcher<G, P>: Default + Matcher<G> {
    /// The constraint type used by the matcher.
    ///
    /// Defines the type of patterns that can be matched by the matcher.
    type Constraint;

    /// Add a pattern to the matcher.
    ///
    /// Patterns are assigned a unique ID, which is returned by this method. All
    /// patterns must be connected.
    fn add_pattern(&mut self, pattern: P) -> PatternID;

    /// Construct a pattern matcher from patterns.
    fn from_patterns(patterns: impl IntoIterator<Item = P>) -> Self
    where
        P: Pattern<Constraint = Self::Constraint>,
    {
        let mut obj = Self::default();
        for pattern in patterns {
            obj.add_pattern(pattern);
        }
        obj
    }
}
