//! The pattern matchers.
//!
//! The `Matcher` trait is the main interface for pattern matching. Implementations
//! of this trait include the [`SinglePatternMatcher`], which matches a single pattern,
//! as well as the [`NaiveManyMatcher`] and [`LineGraphTrie`] types, which match many patterns
//! at once.

pub mod many_patterns;
pub mod single_pattern;

pub use many_patterns::{ManyMatcher, NaiveManyMatcher, PatternID, UnweightedManyMatcher};
use portgraph::PortOffset;
pub use single_pattern::SinglePatternMatcher;

use crate::{Pattern, Property, Universe};

/// A trait for pattern matchers.
///
/// A pattern matcher is a type that can find matches of a pattern in a graph.
/// Implement [`Matcher::find_anchored_matches`] that finds matches of all
/// patterns anchored at a given root node.
pub trait PortMatcher<Graph> {
    /// Node type of the graph.
    type N: Universe;
    /// Node properties
    type PNode;
    /// Edge properties
    type PEdge: Property;

    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_rooted_matches(&self, graph: Graph, root: Self::N) -> Vec<Match<'_, Self, Graph>>;

    /// All the nodes of the graph.
    fn nodes(g: Graph) -> Vec<Self::N>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`PortMatcher::find_anchored_matches`] for each of them.
    fn find_matches(&self, graph: Graph) -> Vec<Match<'_, Self, Graph>>
    where
        Graph: Copy,
    {
        let mut matches = Vec::new();
        for root in Self::nodes(graph) {
            matches.append(&mut self.find_rooted_matches(graph, root));
        }
        matches
    }
}

type Match<'p, M, G> = PatternMatch<
    &'p Pattern<
        <M as PortMatcher<G>>::N,
        <M as PortMatcher<G>>::PNode,
        <M as PortMatcher<G>>::PEdge,
    >,
    <M as PortMatcher<G>>::N,
>;

/// A match instance returned by a Portmatcher instance.
///
/// The PatternID indicates which pattern matches, the root indicates the
/// location of the match, given by the unique mapping of
///                  pattern.root => root
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct PatternMatch<P, N> {
    /// The pattern that matches.
    pub pattern: P,

    /// The root node of the match.
    ///
    /// The entire match can be recovered from the root mapping
    /// using [`crate::Pattern::get_boundary`].
    pub root: N,
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
