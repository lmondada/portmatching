use std::hash::Hash;

use petgraph::visit::{GraphBase, IntoNodeIdentifiers};

use crate::{
    constraint::ScopeConstraint, pattern, ManyMatcher, Pattern, PatternID, SinglePatternMatcher,
};

use super::{
    pattern::UnweightedPortgraphPattern, EdgeProperty, NodeProperty, PortgraphPattern,
};

/// A pattern matcher on portgraphs.
///
/// Loop through every possible root node in the graph and query a root port
/// matcher for each.
///
/// For graphs that implement petgraph's `IntoNodeIdentifiers` trait, it suffices
/// to implement [`RootedPortMatcher`] and rely on the default implementations
/// of [`PortMatcher`].
pub trait PortMatcher: RootedPortMatcher {
    fn nodes_iter(
        &self,
        graph: GraphRef<Self>,
    ) -> impl Iterator<Item = pattern::Value<Self::Pattern>>;

    /// Find matches of all patterns in `graph`.
    ///
    /// The default implementation loops over all possible `root` nodes and
    /// calls [`PortMatcher::find_anchored_matches`] for each of them.
    fn find_matches(&self, graph: GraphRef<Self>) -> Vec<pattern::Match<Self::Pattern>> {
        let mut matches = Vec::new();
        for root in self.nodes_iter(graph) {
            matches.append(&mut self.find_rooted_matches(graph, root));
        }
        matches
    }
}
/// A pattern matcher for fixed root vertices in port graphs
///
/// A pattern matcher is a type that can find matches of a pattern in a graph.
/// Implement [`RootedPortMatcher::find_rooted_matches`] that finds matches of all
/// patterns rooted at a given root node.
pub trait RootedPortMatcher {
    /// Pattern
    type Pattern: Pattern;

    /// Find matches of all patterns in `graph` anchored at the given `root`.
    fn find_rooted_matches(
        &self,
        graph: pattern::DataRef<Self::Pattern>,
        root: pattern::Value<Self::Pattern>,
    ) -> Vec<pattern::Match<Self::Pattern>>;

    fn get_pattern(&self, id: PatternID) -> Option<&Self::Pattern>;
}

type GraphRef<'g, M> = pattern::DataRef<'g, <M as RootedPortMatcher>::Pattern>;
type NodeId<M> = pattern::Value<<M as RootedPortMatcher>::Pattern>;

impl<U: Eq + Hash + Copy + Ord, M: RootedPortMatcher<Pattern = UnweightedPortgraphPattern<U>>>
    PortMatcher for M
where
    for<'g> GraphRef<'g, Self>: IntoNodeIdentifiers + GraphBase<NodeId = NodeId<Self>>,
{
    fn nodes_iter(
        &self,
        graph: GraphRef<Self>,
    ) -> impl Iterator<Item = pattern::Value<Self::Pattern>> {
        graph.node_identifiers()
    }
}

impl<U, PNode, PEdge> RootedPortMatcher for ManyMatcher<PortgraphPattern<U, PNode, PEdge>>
where
    PortgraphPattern<U, PNode, PEdge>: Pattern,
    <PortgraphPattern<U, PNode, PEdge> as Pattern>::Constraint: ScopeConstraint,
    PEdge: EdgeProperty,
    U: Eq + Hash + Copy,
    PNode: NodeProperty,
{
    type Pattern = PortgraphPattern<U, PNode, PEdge>;

    fn find_rooted_matches(
        &self,
        graph: pattern::DataRef<Self::Pattern>,
        root: pattern::Value<Self::Pattern>,
    ) -> Vec<pattern::Match<PortgraphPattern<U, PNode, PEdge>>> {
        self.run(root, graph)
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Self::Pattern> {
        self.get_pattern(id)
    }
}

impl<U> RootedPortMatcher for ManyMatcher<UnweightedPortgraphPattern<U>>
where
    UnweightedPortgraphPattern<U>: Pattern,
    U: Eq + Hash + Copy,
{
    type Pattern = UnweightedPortgraphPattern<U>;

    fn find_rooted_matches(
        &self,
        graph: pattern::DataRef<Self::Pattern>,
        root: pattern::Value<Self::Pattern>,
    ) -> Vec<pattern::Match<Self::Pattern>> {
        self.run(root, graph)
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Self::Pattern> {
        todo!()
    }
}

impl<P: Pattern> RootedPortMatcher for SinglePatternMatcher<P> {
    type Pattern = P;

    fn find_rooted_matches(
        &self,
        graph: pattern::DataRef<Self::Pattern>,
        root: pattern::Value<Self::Pattern>,
    ) -> Vec<pattern::Match<Self::Pattern>> {
        self.find_rooted_match(root, graph)
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Self::Pattern> {
        assert!(id.0 == 0);
        Some(&self.pattern())
    }
}
