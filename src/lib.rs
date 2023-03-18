pub mod matcher;
pub mod pattern;
mod utils;

use std::fmt::Display;

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex};

/// A data layout-independent ID for the edges incident in a node
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum PortOffset {
    Incoming(usize),
    Outgoing(usize),
}

impl Display for PortOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            PortOffset::Incoming(offset) => write!(f, "In{}", offset),
            PortOffset::Outgoing(offset) => write!(f, "Out{}", offset),
        }
    }
}

impl PortOffset {
    fn try_from_index(port: PortIndex, graph: &PortGraph) -> Option<Self> {
        let offset = graph.port_offset(port)?;
        match graph.port_direction(port)? {
            Direction::Incoming => PortOffset::Incoming(offset).into(),
            Direction::Outgoing => PortOffset::Outgoing(offset).into(),
        }
    }

    fn get_index(&self, node: NodeIndex, graph: &PortGraph) -> Option<PortIndex> {
        match *self {
            PortOffset::Incoming(offset) => graph.input(node, offset),
            PortOffset::Outgoing(offset) => graph.output(node, offset),
        }
    }
}
