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
pub trait Matcher<Graph> {
    /// A pattern match as returned by [`Matcher::find_anchored_matches`] and [`Matcher::find_matches`].
    type Match;

    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_anchored_matches(&self, graph: Graph) -> Vec<Self::Match>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`Matcher::find_anchored_matches`] for each of them.
    fn find_matches<'g>(
        &self,
        graph: &'g PortGraph,
    ) -> Vec<<Self as Matcher<(&'g PortGraph, NodeIndex)>>::Match>
    where
        Self: Matcher<(&'g PortGraph, NodeIndex)>,
    {
        let mut matches = Vec::new();
        for root in graph.nodes_iter() {
            matches.append(&mut self.find_anchored_matches((graph, root)));
        }
        matches
    }

    /// Find matches of all patterns in `graph` with `weights`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`Matcher::find_anchored_matches`] for each of them.
    fn find_weighted_matches<'g, W: Copy>(
        &self,
        graph: &'g PortGraph,
        weights: W,
    ) -> Vec<<Self as Matcher<(&'g PortGraph, W, NodeIndex)>>::Match>
    where
        Self: Matcher<(&'g PortGraph, W, NodeIndex)>,
    {
        let mut matches = Vec::new();
        for root in graph.nodes_iter() {
            matches.append(&mut self.find_anchored_matches((graph, weights, root)));
        }
        matches
    }
}
