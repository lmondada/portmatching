use std::borrow::Borrow;

use portgraph::{NodeIndex, PortGraph, PortView};

pub type Node<G> = <G as GraphNodes>::Node;

pub trait GraphNodes {
    type Node;
    type NodesIter<'a>: Iterator<Item = Self::Node>
    where
        Self: 'a;

    fn nodes(&self) -> Self::NodesIter<'_>;
}

impl<G: Borrow<PortGraph>> GraphNodes for G {
    type Node = NodeIndex;
    type NodesIter<'a> = <PortGraph as PortView>::Nodes<'a>
    where Self: 'a;

    fn nodes(&self) -> Self::NodesIter<'_> {
        self.borrow().nodes_iter()
    }
}
