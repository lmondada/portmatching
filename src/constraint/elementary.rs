use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};
use std::fmt::Debug;

use super::{NodeAddress, NodeRange};

/// All constraints can be decomposed into a series of the following
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ElementaryConstraint<N> {
    /// The port must have this label
    PortLabel(PortLabel),
    /// The node weight of the port
    NodeWeight(N),
    /// The port must not match any of these address ranges
    NoMatch(NodeRange),
    /// The port must have this address
    Match(NodeAddress),
}

impl<N: Eq> ElementaryConstraint<N> {
    pub(super) fn is_satisfied(
        &self,
        port: PortIndex,
        graph: &PortGraph,
        root: NodeIndex,
        weight: &N,
    ) -> bool {
        match self {
            ElementaryConstraint::PortLabel(label) => {
                label.is_satisfied(graph.port_offset(port).expect("invalid port"))
            }
            ElementaryConstraint::NodeWeight(w) => weight == w,
            ElementaryConstraint::NoMatch(range) => {
                let node = graph.port_node(port).expect("invalid port");
                range.verify_no_match(node, graph, root)
            }
            ElementaryConstraint::Match(address) => {
                let node = graph.port_node(port).expect("invalid port");
                address.get_node(graph, root) == Some(node)
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum PortLabel {
    Outgoing(usize),
    Incoming(usize),
}

impl PortLabel {
    fn as_port_offset(&self) -> PortOffset {
        match *self {
            PortLabel::Outgoing(offset) => PortOffset::new_outgoing(offset),
            PortLabel::Incoming(offset) => PortOffset::new_incoming(offset),
        }
    }

    pub fn is_satisfied(&self, offset: PortOffset) -> bool {
        offset == self.as_port_offset()
    }

    pub fn and(self, e: PortLabel) -> Option<PortLabel> {
        if self == e {
            Some(self)
        } else {
            None
        }
    }
}

impl Debug for PortLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Outgoing(arg0) => f.write_str(&format!("out({:?})", arg0)),
            Self::Incoming(arg0) => f.write_str(&format!("in({:?})", arg0)),
        }
    }
}
