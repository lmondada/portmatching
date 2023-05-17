//! The pattern matchers.
//!
//! The `Matcher` trait is the main interface for pattern matching. Implementations
//! of this trait include the [`SinglePatternMatcher`], which matches a single pattern,
//! as well as the [`NaiveManyMatcher`] and [`LineGraphTrie`] types, which match many patterns
//! at once.
use portgraph::{NodeIndex, PortGraph};

pub mod many_patterns;
pub mod single_pattern;

pub use many_patterns::{
    ManyPatternMatcher, NaiveManyMatcher, PatternID, TrieConstruction, TrieMatcher,
};
pub use single_pattern::SinglePatternMatcher;

/// A trait for pattern matchers.
///
/// A pattern matcher is a type that can find matches of a pattern in a graph.
/// Implement [`Matcher::find_anchored_matches`] that finds matches of all
/// patterns anchored at a given root node.
pub trait Matcher {
    /// A pattern match as returned by [`Matcher::find_anchored_matches`] and [`Matcher::find_matches`].
    type Match;
    /// The graph type that the matcher operates on.
    type Graph<'g>;

    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_anchored_matches<'g>(&self, graph: Self::Graph<'g>) -> Vec<Self::Match>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`Matcher::find_anchored_matches`] for each of them.
    fn find_matches<'g>(&self, graph: &'g PortGraph) -> Vec<Self::Match>
    where
        Self: Matcher<Graph<'g> = (&'g PortGraph, NodeIndex)>,
    {
        let mut matches = Vec::new();
        for root in graph.nodes_iter() {
            matches.append(&mut self.find_anchored_matches((graph, root)));
        }
        matches
    }
}
