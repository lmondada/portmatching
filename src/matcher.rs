//! The pattern matchers.
//!
//! The [`PortMatcher`] trait is the main interface for pattern matching. The
//! following implementations of this trait are provided:
//!  - [`SinglePatternMatcher`], which matches a single pattern,
//!  - [`NaiveManyMatcher`], matching one pattern at a time using
//!    [`SinglePatternMatcher`]. Mostly useful as a benchmark and for testing.
//!  - [`ManyMatcher`], which match many patterns at once. The main matcher
//!    implementation of this crate.

mod many_patterns;
mod single_pattern;

use derive_more::{From, Into};
use std::{fmt::Debug, hash::Hash};

pub use self::many_patterns::{ManyMatcher, NaiveManyMatcher};
pub use self::single_pattern::SinglePatternMatcher;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Identify patterns with IDs.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default, From, Into, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PatternID(pub usize);

/// Match patterns on host data `D`.
pub trait PortMatcher<D> {
    /// The data returned by the matcher alongside pattern IDs.
    type Match;

    /// Find matches of all patterns in `host``.
    ///
    /// Matches are expressed by a [`PatternID`] and arbitrary match data of
    /// type [`PortMatcher::Match`].
    fn find_matches<'a>(
        &'a self,
        host: &'a D,
    ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a;
}

/// A match instance returned by a Portmatcher instance.
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PatternMatch<M> {
    /// The matching pattern ID.
    pub pattern: PatternID,

    /// Match data such as match position in the host.
    pub match_data: M,
}

impl<M> PatternMatch<M> {
    /// Create a new pattern match result.
    pub fn new(pattern: PatternID, match_data: M) -> Self {
        Self {
            pattern,
            match_data,
        }
    }
}

impl<M> From<(PatternID, M)> for PatternMatch<M> {
    fn from((pattern, match_data): (PatternID, M)) -> Self {
        Self::new(pattern, match_data)
    }
}
