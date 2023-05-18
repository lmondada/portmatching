use std::iter::repeat;

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::utils::{follow_path, port_opposite};

use super::{NodeAddress, NodeRange};

/// All constraints can be decomposed into a series of the following
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
                verify_no_match(node, range, graph, root)
            }
            ElementaryConstraint::Match(address) => {
                let node = graph.port_node(port).expect("invalid port");
                address.get_node(graph, root) == Some(node)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

/// Whether node does not appear in the address range
fn verify_no_match(
    node: NodeIndex,
    NodeRange { spine, range }: &NodeRange,
    g: &PortGraph,
    root: NodeIndex,
) -> bool {
    let Some(root) = follow_path(&spine.path, root, g) else {
        return true
    };
    if root == node {
        return false;
    }

    let n_neg = -range.start() as usize;
    let n_pos = if range.end() >= 0 {
        range.end() as usize
    } else {
        0
    };

    // go in both directions from root
    for (port, n_jumps) in [
        (g.output(root, spine.offset), n_pos),
        (g.input(root, spine.offset), n_neg),
    ] {
        let Some(port) = port else { continue };
        if n_times(n_jumps)
            .scan(port, |port, ()| {
                let next_port = g.port_link(*port)?;
                let node = g.port_node(next_port).expect("invalid port");
                *port = port_opposite(next_port, g)?;
                Some(node)
            })
            .any(|in_range| node == in_range)
        {
            return false;
        }
    }
    true
}

fn n_times(n: usize) -> impl Iterator<Item = ()> {
    repeat(()).take(n)
}
