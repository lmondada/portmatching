//! The pattern matchers.
//!
//! The `Matcher` trait is the main interface for pattern matching. Implementations
//! of this trait include the [`SinglePatternMatcher`], which matches a single pattern,
//! as well as the [`NaiveManyMatcher`] and [`LineGraphTrie`] types, which match many patterns
//! at once.

pub mod many_patterns;
pub mod single_pattern;

use std::hash::Hash;

pub use many_patterns::{ManyMatcher, NaiveManyMatcher, PatternID};
pub use single_pattern::SinglePatternMatcher;

use crate::{pattern, portgraph::RootedPortMatcher, HashMap, Pattern, ScopeConstraint};

/// A match instance returned by a Portmatcher instance.
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PatternMatch<P, V> {
    /// The pattern that matches.
    pub pattern: P,

    /// The root node of the match in the host graph
    pub root: V,
}

impl<P, V> PatternMatch<P, V> {
    pub fn new(pattern: P, root: V) -> Self {
        Self { pattern, root }
    }

    pub fn from_tuple((pattern, root): (P, V)) -> Self {
        Self { pattern, root }
    }
}

impl<'p, P, V> PatternMatch<&'p P, V> {
    pub fn clone_pattern(&self) -> PatternMatch<P, V>
    where
        P: Clone,
        V: Copy,
    {
        PatternMatch {
            pattern: self.pattern.clone(),
            root: self.root,
        }
    }
}

impl<P, V: Copy> PatternMatch<P, V> {
    pub fn as_ref(&self) -> PatternMatch<&P, V> {
        PatternMatch {
            pattern: &self.pattern,
            root: self.root,
        }
    }

    // pub fn to_match_map<G: LinkView + Copy>(&self, graph: G) -> Option<patterns::ScopeMap<P>> {
    //     self.as_ref().to_match_map(graph)
    // }
}

impl<'p, P: Pattern + Clone> PatternMatch<&'p P, pattern::Value<P>> {
    pub fn to_match_map(
        &self,
        graph: pattern::DataRef<P>,
    ) -> Option<HashMap<P::Universe, pattern::Value<P>>> {
        Some(
            SinglePatternMatcher::new(self.pattern.clone())
                .get_match_map(self.root, graph)?
                .into_iter()
                .collect(),
        )
    }
}

impl<V: Copy> PatternMatch<PatternID, V> {
    pub fn to_match_map<P: Pattern + Clone>(
        &self,
        graph: pattern::DataRef<P>,
        matcher: &impl RootedPortMatcher<Pattern = P>,
    ) -> Option<HashMap<P::Universe, pattern::Value<P>>>
    where
        P::Constraint: ScopeConstraint<Value = V>,
    {
        PatternMatch::new(matcher.get_pattern(self.pattern)?, self.root).to_match_map(graph)
    }
}
