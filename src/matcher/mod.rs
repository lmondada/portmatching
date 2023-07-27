//! The pattern matchers.
//!
//! The `Matcher` trait is the main interface for pattern matching. Implementations
//! of this trait include the [`SinglePatternMatcher`], which matches a single pattern,
//! as well as the [`NaiveManyMatcher`] and [`LineGraphTrie`] types, which match many patterns
//! at once.

pub mod many_patterns;
pub mod single_pattern;

use std::hash::Hash;

pub use many_patterns::{ManyMatcher, NaiveManyMatcher, PatternID, UnweightedManyMatcher};
use petgraph::visit::{GraphBase, IntoNodeIdentifiers};
use portgraph::{LinkView, NodeIndex};
pub use single_pattern::SinglePatternMatcher;

use crate::{
    patterns::{Edge, UnweightedEdge},
    HashMap, NodeProperty, Pattern, Universe,
};

use self::single_pattern::validate_unweighted_edge;

/// A trait for pattern matchers.
///
/// A pattern matcher is a type that can find matches of a pattern in a graph.
/// Implement [`Matcher::find_anchored_matches`] that finds matches of all
/// patterns anchored at a given root node.
pub trait PortMatcher<GraphRef, NodeId, U: Universe> {
    /// Node properties
    type PNode;
    /// Edge properties
    type PEdge: Eq + Hash;

    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_rooted_matches(&self, graph: GraphRef, root: NodeId) -> Vec<Match>;

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, Self::PNode, Self::PEdge>>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`PortMatcher::find_anchored_matches`] for each of them.
    fn find_matches(&self, graph: GraphRef) -> Vec<Match>
    where
        GraphRef: IntoNodeIdentifiers + GraphBase<NodeId = NodeId>,
    {
        let mut matches = Vec::new();
        for root in graph.node_identifiers() {
            matches.append(&mut self.find_rooted_matches(graph, root));
        }
        matches
    }
}

type Match = PatternMatch<PatternID, NodeIndex>;

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

impl<U: Universe, PNode: NodeProperty> PatternMatch<Pattern<U, PNode, UnweightedEdge>, NodeIndex> {
    pub fn as_ref(&self) -> PatternMatch<&Pattern<U, PNode, UnweightedEdge>, NodeIndex> {
        PatternMatch {
            pattern: &self.pattern,
            root: self.root,
        }
    }

    pub fn to_match_map<G: LinkView + Copy>(&self, graph: G) -> Option<HashMap<U, NodeIndex>> {
        self.as_ref().to_match_map(graph)
    }
}

impl<'p, U: Universe, PNode: NodeProperty>
    PatternMatch<&'p Pattern<U, PNode, UnweightedEdge>, NodeIndex>
{
    pub fn to_match_map<G: LinkView + Copy>(&self, graph: G) -> Option<HashMap<U, NodeIndex>> {
        Some(
            SinglePatternMatcher::from_pattern(self.pattern.clone())
                .get_match_map(
                    graph,
                    self.root,
                    forget_node_weight(validate_unweighted_edge),
                )?
                .into_iter()
                .collect(),
        )
    }
}

impl PatternMatch<PatternID, NodeIndex> {
    pub fn to_match_map<G, M, U>(&self, graph: G, matcher: &M) -> Option<HashMap<U, NodeIndex>>
    where
        G: LinkView + Copy,
        M::PNode: NodeProperty,
        M: PortMatcher<G, NodeIndex, U, PEdge = UnweightedEdge>,
        U: Universe,
    {
        PatternMatch::new(matcher.get_pattern(self.pattern)?, self.root).to_match_map(graph)
    }
}

fn forget_node_weight<W, G, F>(
    f: F,
) -> impl for<'a> Fn(Edge<NodeIndex, &'a W, &'a UnweightedEdge>, G) -> Option<(NodeIndex, NodeIndex)>
where
    F: for<'a> Fn(Edge<NodeIndex, &'a (), &'a UnweightedEdge>, G) -> Option<(NodeIndex, NodeIndex)>,
{
    move |e, g| {
        let Edge {
            source,
            target,
            edge_prop,
            ..
        } = e;
        f(
            Edge {
                source,
                target,
                edge_prop,
                source_prop: None,
                target_prop: None,
            },
            g,
        )
    }
}
