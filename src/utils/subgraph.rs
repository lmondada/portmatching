//! A view into a petgraph subgraph.

use std::hash::Hash;

use petgraph::{
    visit::{GraphBase, GraphRef, IntoNeighbors, IntoNeighborsDirected},
    Direction,
};

use crate::HashSet;

pub(crate) struct SubgraphRef<'g, G, NodeId> {
    graph: &'g G,
    nodes: &'g HashSet<NodeId>,
    reversed: bool,
}

impl<'g, G, NodeId> Clone for SubgraphRef<'g, G, NodeId> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'g, G, NodeId> Copy for SubgraphRef<'g, G, NodeId> {}

impl<'g, G> SubgraphRef<'g, G, <&'g G as GraphBase>::NodeId>
where
    &'g G: GraphBase,
{
    /// Create a new subgraph reference.
    pub(crate) fn new(
        graph: &'g G,
        nodes: &'g HashSet<<&'g G as GraphBase>::NodeId>,
        reversed: bool,
    ) -> Self {
        Self {
            graph,
            nodes,
            reversed,
        }
    }
}

impl<'g, G> GraphBase for SubgraphRef<'g, G, <&'g G as GraphBase>::NodeId>
where
    &'g G: GraphBase,
{
    type EdgeId = <&'g G as GraphBase>::EdgeId;

    type NodeId = <&'g G as GraphBase>::NodeId;
}

impl<'g, G> GraphRef for SubgraphRef<'g, G, <&'g G as GraphBase>::NodeId> where &'g G: GraphBase {}

impl<'g, G> IntoNeighbors for SubgraphRef<'g, G, <&'g G as GraphBase>::NodeId>
where
    &'g G: IntoNeighbors + GraphBase,
    <&'g G as GraphBase>::NodeId: Hash + Eq,
{
    type Neighbors = Box<dyn Iterator<Item = Self::NodeId> + 'g>;

    fn neighbors(self, n: Self::NodeId) -> Self::Neighbors {
        Box::new(self.graph.neighbors(n).filter(|n| self.nodes.contains(n)))
    }
}

impl<'g, G> IntoNeighborsDirected for SubgraphRef<'g, G, <&'g G as GraphBase>::NodeId>
where
    &'g G: IntoNeighborsDirected + GraphBase,
    <&'g G as GraphBase>::NodeId: Hash + Eq,
{
    type NeighborsDirected = Box<dyn Iterator<Item = Self::NodeId> + 'g>;

    fn neighbors_directed(self, n: Self::NodeId, mut d: Direction) -> Self::NeighborsDirected {
        if self.reversed {
            d = d.opposite();
        }

        Box::new(
            self.graph
                .neighbors_directed(n, d)
                .filter(|n| self.nodes.contains(n)),
        )
    }
}
