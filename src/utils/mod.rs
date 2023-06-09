//! Utility functions.
//!
//! Much of this should probably be moved upstream to `portgraph`.

pub(crate) mod cover;
mod depth;
// pub(crate) mod iter;
pub(crate) mod pre_order;

pub(crate) mod zero_range;
pub(crate) use zero_range::ZeroRange;

pub use depth::is_connected;
pub(crate) use depth::{centre, NoCentreError};

pub(crate) mod toposort;

pub(crate) mod age;

#[cfg(test)]
pub(crate) mod test_utils;

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

/// Returns the port on the opposite side of the same node
pub(crate) fn port_opposite(port: PortIndex, graph: &PortGraph) -> Option<PortIndex> {
    let node = graph.port_node(port).expect("invalid port");
    let offset = graph.port_offset(port).expect("invalid port");
    let offset_op = match offset.direction() {
        Direction::Incoming => PortOffset::new_outgoing(offset.index()),
        Direction::Outgoing => PortOffset::new_incoming(offset.index()),
    };
    graph.port_index(node, offset_op)
}

/// Follow path from root node
pub(crate) fn follow_path(
    path: &[PortOffset],
    root: NodeIndex,
    graph: &PortGraph,
) -> Option<NodeIndex> {
    let mut curr_node = root;
    for &offset in path {
        let out_port = graph.port_index(curr_node, offset)?;
        let in_port = graph.port_link(out_port)?;
        curr_node = graph.port_node(in_port)?;
    }
    Some(curr_node)
}
