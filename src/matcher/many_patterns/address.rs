use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Display},
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AddressOrRoot {
    Address(Box<NodeAddress>),
    Root,
}

/// The address of a node in a graph
///
/// A way to identify a node uniquely in a connected pattern/graph,
/// independently of the numbering of the vertices
///
/// This works by, in essence, saving a unique path to be followed from a fixed
/// root in the graph.
#[derive(PartialEq, Eq, Clone, Debug)]
pub(crate) struct NodeAddress {
    line_start: AddressOrRoot,
    out_port: PortOffset,
    line_index: usize,
}

impl Display for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.depth(), self.line_index)
    }
}

impl NodeAddress {
    fn depth(&self) -> usize {
        match &self.line_start {
            AddressOrRoot::Address(addr) => 1 + addr.depth(),
            AddressOrRoot::Root => 0,
        }
    }

    /// The address of `node` with respect to `root`
    pub(crate) fn new(node: NodeIndex, graph: &PortGraph, root: NodeIndex) -> Self {
        let (node2addr, root2port) = {
            let mut node2addr = BTreeMap::from([(root, (0, 0))]);
            let mut root2port = Vec::new();
            let mut visited_ports = BTreeSet::new();
            let mut ports_to_visit = VecDeque::new();
            ports_to_visit.extend(graph.inputs(root).chain(graph.outputs(root)));
            let mut line_cnt = 0;
            while let Some(mut port) = ports_to_visit.pop_front() {
                if visited_ports.contains(&port) {
                    continue;
                }
                root2port.push(port);
                let mut line_ind = 1;
                loop {
                    visited_ports.insert(port);
                    let Some(in_port) = graph.port_link(port) else { break };
                    visited_ports.insert(in_port);
                    let Some(node) = graph.port_node(in_port) else { break };
                    if !node2addr.contains_key(&node) {
                        // Discovered a new node
                        node2addr.insert(node, (line_cnt, line_ind));
                        ports_to_visit.extend(graph.inputs(node).chain(graph.outputs(node)));
                    }
                    let in_offset = graph.port_offset(in_port).expect("invalid port");
                    let out_offset = port_opposite(in_offset);
                    let Some(out_port) = graph.port_index(node, out_offset) else { break };
                    if visited_ports.contains(&out_port) {
                        break;
                    } else {
                        port = out_port;
                        line_ind += 1;
                    }
                }
                line_cnt += 1;
            }
            (node2addr, root2port)
        };

        // backtrack to find path to root
        fn create_address(
            node: NodeIndex,
            graph: &PortGraph,
            root: NodeIndex,
            addresses: &BTreeMap<NodeIndex, (usize, usize)>,
            roots: &Vec<PortIndex>,
        ) -> NodeAddress {
            let (line_cnt, line_index) = addresses[&node];
            if (line_cnt, line_index) == (0, 0) {
                return NodeAddress {
                    line_start: AddressOrRoot::Root,
                    out_port: PortOffset::new_outgoing(0),
                    line_index: 0,
                };
            }
            let node = graph.port_node(roots[line_cnt]).expect("invalid port");
            let line_start = if line_cnt == 0 {
                AddressOrRoot::Root
            } else {
                let addr = create_address(node, graph, root, addresses, roots);
                if addr.is_root() {
                    AddressOrRoot::Root
                } else {
                    AddressOrRoot::Address(Box::new(addr))
                }
            };
            let out_port = graph.port_offset(roots[line_cnt]).expect("invalid port");
            NodeAddress {
                line_start,
                out_port,
                line_index,
            }
        }
        create_address(node, graph, root, &node2addr, &root2port)
    }

    fn is_root(&self) -> bool {
        self.line_start == AddressOrRoot::Root && self.line_index == 0
    }

    /// Get NodeIndex at this address for `graph`
    ///
    /// Note that addresses are always relative to the `root`
    pub fn get_node(&self, graph: &PortGraph, root: NodeIndex) -> Option<NodeIndex> {
        let line_start = match &self.line_start {
            AddressOrRoot::Address(addr) => addr.get_node(graph, root)?,
            AddressOrRoot::Root => root,
        };
        if self.line_index == 0 {
            return Some(line_start);
        }
        let mut node = line_start;
        let mut out_port = graph.port_index(node, self.out_port);
        for _ in 0..self.line_index {
            let in_port = graph.port_link(out_port?)?;
            node = graph.port_node(in_port)?;
            let offset = graph.port_offset(in_port).expect("invalid port");
            out_port = graph.port_index(node, port_opposite(offset));
        }
        Some(node)
    }
}

fn port_opposite(offset: PortOffset) -> PortOffset {
    match offset.direction() {
        Direction::Incoming => PortOffset::new_outgoing(offset.index()),
        Direction::Outgoing => PortOffset::new_incoming(offset.index()),
    }
}

#[cfg(test)]
mod tests {
    use super::NodeAddress;
    use crate::utils::test_utils::gen_portgraph_connected;
    use portgraph::proptest::gen_node_index;
    use portgraph::NodeIndex;

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn create_address((g, node) in gen_node_index(gen_portgraph_connected(10, 4, 20))) {
            let root = NodeIndex::new(0);
            let addr = NodeAddress::new(node, &g, root);
            prop_assert_eq!(addr.get_node(&g, root).unwrap(), node);
        }
    }
}
