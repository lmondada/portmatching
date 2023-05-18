use std::collections::{BTreeSet, VecDeque};

use portgraph::{NodeIndex, PortGraph, PortIndex};

use crate::{
    constraint::UnweightedAdjConstraint,
    utils::{centre, NoCentreError},
    Skeleton,
};

use super::{Edge, InvalidPattern, Pattern};

/// A pattern graph.
///
/// Patterns must be connected and have a fixed `root` node,
/// which by default is chosen to be the centre of the graph, for fast
/// matching and short relative paths to the root.
#[derive(Debug, Clone)]
pub struct UnweightedPattern {
    /// The pattern graph.
    pub(crate) graph: PortGraph,
    /// The root of the pattern.
    pub(crate) root: NodeIndex,
}

impl Pattern for UnweightedPattern {
    type Constraint = UnweightedAdjConstraint;

    fn graph(&self) -> &PortGraph {
        &self.graph
    }

    fn root(&self) -> NodeIndex {
        self.root
    }

    fn to_constraint(&self, Edge(_, in_port): &Edge) -> Self::Constraint {
        if let &Some(in_port) = in_port {
            // TODO: dont recompute skeleton each time
            let skeleton = Skeleton::new(&self.graph, self.root);
            let (label, addr, no_addr) = skeleton.get_coordinates(in_port);
            UnweightedAdjConstraint::link(label, addr, no_addr)
        } else {
            UnweightedAdjConstraint::dangling()
        }
    }

    fn all_lines(&self) -> Vec<Vec<Edge>> {
        let mut node_queue = VecDeque::from([self.root]);
        let mut all_lines = Vec::new();
        let mut visited_ports = BTreeSet::new();
        while let Some(node) = node_queue.pop_front() {
            while let Some(port) = self
                .graph
                .all_ports(node)
                .find(|p| !visited_ports.contains(p))
            {
                let line = get_line(&self.graph, port, &mut visited_ports);
                for new_node in line
                    .iter()
                    .filter_map(|e| e.1.and_then(|p| self.graph.port_node(p)))
                {
                    node_queue.push_back(new_node);
                }
                all_lines.push(line);
            }
        }
        all_lines
    }
}

impl UnweightedPattern {
    /// Create a new pattern from a graph.
    pub fn from_graph(graph: PortGraph) -> Result<Self, InvalidPattern> {
        let root = centre(&graph).map_err(|err| match err {
            NoCentreError::DisconnectedGraph => InvalidPattern::DisconnectedPattern,
            NoCentreError::EmptyGraph => InvalidPattern::EmptyPattern,
        })?;
        Ok(UnweightedPattern { graph, root })
    }
}

/// Starting at `port`, keep following edges as long as possible
fn get_line(
    graph: &PortGraph,
    mut port: PortIndex,
    visited_ports: &mut BTreeSet<PortIndex>,
) -> Vec<Edge> {
    let mut ports = Vec::new();
    loop {
        let link_port = graph.port_link(port);
        ports.push(Edge(port, link_port));
        visited_ports.insert(port);
        if let Some(other_port) = link_port {
            visited_ports.insert(other_port);
        }
        if let Some(new_port) = link_port.and_then(|link_port| traverse_node(graph, link_port)) {
            port = new_port;
        } else {
            break;
        }
        if visited_ports.contains(&port) {
            break;
        }
    }
    ports
}

/// For every n-th in/output port, return n-th out/input port
fn traverse_node(graph: &PortGraph, port: PortIndex) -> Option<PortIndex> {
    let node = graph.port_node(port)?;
    let in_init = graph.input(node, 0)?;
    let out_init = graph.output(node, 0)?;
    if port.index() >= out_init.index() {
        let offset = port.index() - out_init.index();
        graph.input(node, offset)
    } else {
        let offset = port.index() - in_init.index();
        graph.output(node, offset)
    }
}

#[cfg(test)]
mod tests {
    use crate::{pattern::InvalidPattern, utils::test_utils::*};

    use super::*;
    use itertools::Itertools;
    use portgraph::{PortGraph, PortOffset};

    #[test]
    fn empty_pattern() {
        let g = PortGraph::new();
        assert_eq!(
            UnweightedPattern::from_graph(g).unwrap_err(),
            InvalidPattern::EmptyPattern
        );
    }

    #[test]
    fn ok_pattern() {
        let mut g = PortGraph::new();
        g.add_node(3, 3);
        UnweightedPattern::from_graph(g).unwrap();
    }

    #[test]
    fn edge_order() {
        let g = graph();
        let (v0_ports, v1_ports, v2_ports, v3_ports) = g
            .nodes_iter()
            .map(|n| g.all_ports(n).collect_vec())
            .collect_tuple()
            .unwrap();
        let p = UnweightedPattern::from_graph(g).unwrap();
        assert_eq!(
            p.canonical_edge_ordering(),
            vec![
                Edge(v2_ports[0], v1_ports[3].into()),
                Edge(v1_ports[1], None),
                Edge(v2_ports[1], v0_ports[2].into()),
                Edge(v0_ports[0], None),
                Edge(v2_ports[2], v3_ports[1].into()),
                Edge(v3_ports[3], None),
                Edge(v1_ports[0], None),
                Edge(v1_ports[2], None),
                Edge(v1_ports[4], None),
                Edge(v0_ports[1], None),
                Edge(v0_ports[3], None),
                Edge(v3_ports[0], None),
                Edge(v3_ports[2], None)
            ]
        );
    }

    #[test]
    fn get_line_test_1() {
        let mut g = graph();
        let v2 = g.nodes_iter().nth(2).unwrap();
        let v3 = g.nodes_iter().nth(3).unwrap();
        let v4 = g.add_node(1, 0);
        let v2_out0 = g.port_index(v2, PortOffset::new_outgoing(0)).unwrap();
        let v4_in0 = g.port_index(v4, PortOffset::new_incoming(0)).unwrap();
        let v3_out0 = g.port_index(v3, PortOffset::new_outgoing(0)).unwrap();
        let v3_in1 = g.port_index(v3, PortOffset::new_incoming(1)).unwrap();
        let v3_out1 = g.port_index(v3, PortOffset::new_outgoing(1)).unwrap();
        g.link_ports(v3_out0, v4_in0).unwrap();
        assert_eq!(
            get_line(&g, v2_out0, &mut BTreeSet::new()),
            vec![Edge(v2_out0, v3_in1.into()), Edge(v3_out1, None)]
        );
    }

    #[test]
    fn get_line_test_2() {
        let mut g = graph();
        let v0 = g.nodes_iter().next().unwrap();
        let v2 = g.nodes_iter().nth(2).unwrap();
        let v3 = g.nodes_iter().nth(3).unwrap();
        let v4 = g.add_node(1, 0);
        let v0_in0 = g.port_index(v0, PortOffset::new_incoming(0)).unwrap();
        let v0_out0 = g.port_index(v0, PortOffset::new_outgoing(0)).unwrap();
        let v2_out0 = g.port_index(v2, PortOffset::new_outgoing(0)).unwrap();
        let v2_in1 = g.port_index(v2, PortOffset::new_incoming(1)).unwrap();
        let v4_in0 = g.port_index(v4, PortOffset::new_incoming(0)).unwrap();
        // let v4_out0 = g.port_index(v4, 0, portgraph::Direction::Incoming).unwrap();
        let v3_in1 = g.port_index(v3, PortOffset::new_incoming(1)).unwrap();
        let v3_out1 = g.port_index(v3, PortOffset::new_outgoing(1)).unwrap();
        g.link_ports(v3_out1, v4_in0).unwrap();
        assert_eq!(
            get_line(&g, v2_out0, &mut BTreeSet::new()),
            vec![Edge(v2_out0, v3_in1.into()), Edge(v3_out1, v4_in0.into())]
        );
        assert_eq!(
            get_line(&g, v2_in1, &mut BTreeSet::new()),
            vec![Edge(v2_in1, v0_out0.into()), Edge(v0_in0, None)]
        );
    }
}
