//! The pattern matchers.
//!
//! The `Matcher` trait is the main interface for pattern matching. Implementations
//! of this trait include the [`SinglePatternMatcher`], which matches a single pattern,
//! as well as the [`NaiveManyMatcher`] and [`LineGraphTrie`] types, which match many patterns
//! at once.

pub mod many_patterns;
pub mod single_pattern;

use derive_more::{From, Into};
use std::{fmt::Debug, hash::Hash};

pub use self::many_patterns::{ManyMatcher, NaiveManyMatcher};
pub use self::single_pattern::SinglePatternMatcher;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// ID for a pattern in a [`ManyPatternMatcher`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct PatternID(pub usize);

type Match<U> = PatternMatch<PatternID, U>;

/// A trait for pattern matchers.
///
/// A pattern matcher is a type that can find matches of a pattern in a graph.
/// Implement [`Matcher::find_anchored_matches`] that finds matches of all
/// patterns anchored at a given root node.
pub trait PortMatcher<U, D> {
    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_rooted_matches(&self, root: U, graph: &D) -> Vec<Match<U>>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`PortMatcher::find_anchored_matches`] for each of them.
    fn find_matches(&self, graph: &D) -> Vec<PatternMatch<PatternID, U>>
    where
        D: IterRootCandidates<U = U>,
    {
        let mut matches = Vec::new();
        for root in graph.root_candidates() {
            matches.append(&mut self.find_rooted_matches(root, graph));
        }
        matches
    }
}

/// A trait for iterating over all root candidates for port matching.
pub trait IterRootCandidates {
    /// The type of the root candidates.
    type U;

    /// Iterate over the root candidates.
    fn root_candidates(&self) -> impl Iterator<Item = Self::U>;
}

/// A trait for variable names.
pub trait VariableNaming: Debug + Hash + Eq + Clone {
    /// Return the name of the root variable.
    fn root_variable() -> Self;
}

/// A heuristic whether a set of constraints should be turned into a deterministic
/// transition.
pub trait DetHeuristic
where
    Self: Sized,
{
    fn make_det<'c>(constraints: &[&'c Self]) -> bool;
}

/// A match instance returned by a Portmatcher instance.
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PatternMatch<P, N> {
    /// The pattern that matches.
    pub pattern: P,

    /// The root node of the match in the host graph
    pub root: N,
}

impl<P, N> PatternMatch<P, N> {
    pub fn new(pattern: P, root: N) -> Self {
        Self { pattern, root }
    }

    pub fn from_tuple((pattern, root): (P, N)) -> Self {
        Self { pattern, root }
    }
}

impl<'p, P, N> PatternMatch<&'p P, N> {
    pub fn clone_pattern(&self) -> PatternMatch<P, N>
    where
        P: Clone,
        N: Copy,
    {
        PatternMatch {
            pattern: self.pattern.clone(),
            root: self.root,
        }
    }
}
