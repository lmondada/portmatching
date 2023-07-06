use portgraph::{NodeIndex, PortGraph, PortView};

pub type Node<G> = <G as GraphNodes>::Node;

pub trait GraphNodes {
    type Node;
    type NodesIter<'a>: Iterator<Item = Self::Node>
    where
        Self: 'a;

    fn nodes(&self) -> Self::NodesIter<'_>;
}

impl GraphNodes for PortGraph {
    type Node = NodeIndex;
    type NodesIter<'a> = <PortGraph as PortView>::Nodes<'a>
    where Self: 'a;

    fn nodes(&self) -> Self::NodesIter<'_> {
        self.nodes_iter()
    }
}

impl<W> GraphNodes for (PortGraph, W) {
    type Node = NodeIndex;
    type NodesIter<'a> = <PortGraph as PortView>::Nodes<'a>
    where Self: 'a;

    fn nodes(&self) -> Self::NodesIter<'_> {
        self.0.nodes_iter()
    }
}
