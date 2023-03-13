use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::utils::{centre, NoCentreError};
use portgraph::{portgraph::PortOffset, NodeIndex, PortGraph, PortIndex};

#[derive(Debug, Clone)]
pub struct Pattern {
    pub(crate) graph: PortGraph,
    pub(crate) root: NodeIndex,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Edge(pub(crate) PortIndex, pub(crate) Option<PortIndex>);

impl Pattern {
    pub fn from_graph(graph: PortGraph) -> Result<Self, InvalidPattern> {
        let root = centre(&graph).map_err(|err| match err {
            NoCentreError::DisconnectedGraph => InvalidPattern::DisconnectedPattern,
            NoCentreError::EmptyGraph => InvalidPattern::EmptyPattern,
        })?;
        Ok(Pattern { graph, root })
    }

    /// Every pattern has a unique canonical ordering of its edges.
    /// Pattern matching can be done by matching these edges one-by-one
    pub(crate) fn canonical_edge_ordering(&self) -> Vec<Edge> {
        self.all_lines().into_iter().flatten().collect()
    }

    pub(crate) fn all_lines(&self) -> Vec<Vec<Edge>> {
        let mut node_queue = VecDeque::from([self.root]);
        let mut all_lines = Vec::new();
        let mut visited_ports = BTreeSet::new();
        while let Some(node) = node_queue.pop_front() {
            while let Some(port) = self
                .graph
                .all_ports(node)
                .filter(|p| !visited_ports.contains(p))
                .next()
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

    pub fn get_boundary(&self, root: NodeIndex, graph: &PortGraph) -> PatternBoundaries {
        let mut out_edges = Vec::new();
        let mut in_edges = Vec::new();
        let mut pattern_to_graph = BTreeMap::from([(self.root, root)]);
        for line in self.all_lines() {
            for edge in line {
                let curr_pattern = self.graph.port_node(edge.0).unwrap();
                let curr_graph = pattern_to_graph[&curr_pattern];
                let port_offset = self.graph.port_index(edge.0).unwrap();
                let port_out = graph.port_alt(curr_graph, port_offset.clone()).unwrap();
                if let Some(next_port) = edge.1 {
                    let next_pattern = self.graph.port_node(next_port).unwrap();
                    let port_in = graph.port_link(port_out).unwrap();
                    let next_graph = graph.port_node(port_in).unwrap();
                    pattern_to_graph.insert(next_pattern, next_graph);
                } else {
                    match &port_offset {
                        PortOffset::InPort(_) => in_edges.push(port_out),
                        PortOffset::OutPort(_) => out_edges.push(port_out),
                    }
                }
            }
        }
        PatternBoundaries {
            _in_edges: in_edges,
            _out_edges: out_edges,
        }
    }
}

#[derive(Debug)]
pub struct PatternBoundaries {
    _in_edges: Vec<PortIndex>,
    _out_edges: Vec<PortIndex>,
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

#[derive(Debug, PartialEq, Eq)]
pub enum InvalidPattern {
    DisconnectedPattern,
    EmptyPattern,
}

#[cfg(test)]
mod tests {
    use crate::utils::test_utils::*;

    use super::*;
    use itertools::Itertools;
    use portgraph::PortGraph;

    #[test]
    fn empty_pattern() {
        let g = PortGraph::new();
        assert_eq!(
            Pattern::from_graph(g).unwrap_err(),
            InvalidPattern::EmptyPattern
        );
    }

    #[test]
    fn ok_pattern() {
        let mut g = PortGraph::new();
        g.add_node(3, 3);
        Pattern::from_graph(g).unwrap();
    }

    #[test]
    fn edge_order() {
        let g = graph();
        let (v0_ports, v1_ports, v2_ports, v3_ports) = g
            .nodes_iter()
            .map(|n| g.all_ports(n).collect_vec())
            .collect_tuple()
            .unwrap();
        let p = Pattern::from_graph(g).unwrap();
        assert_eq!(
            p.canonical_edge_ordering(),
            vec![
                Edge(v2_ports[0].into(), v1_ports[3].into()),
                Edge(v1_ports[1].into(), None),
                Edge(v2_ports[1].into(), v0_ports[2].into()),
                Edge(v0_ports[0].into(), None),
                Edge(v2_ports[2].into(), v3_ports[1].into()),
                Edge(v3_ports[3].into(), None),
                Edge(v1_ports[0].into(), None),
                Edge(v1_ports[2].into(), None),
                Edge(v1_ports[4].into(), None),
                Edge(v0_ports[1].into(), None),
                Edge(v0_ports[3].into(), None),
                Edge(v3_ports[0].into(), None),
                Edge(v3_ports[2].into(), None)
            ]
        );
    }

    #[test]
    fn get_line_test_1() {
        let mut g = graph();
        let v2 = g.nodes_iter().nth(2).unwrap();
        let v3 = g.nodes_iter().nth(3).unwrap();
        let v4 = g.add_node(1, 0);
        let v2_out0 = g.port(v2, 0, portgraph::Direction::Outgoing).unwrap();
        let v4_in0 = g.port(v4, 0, portgraph::Direction::Incoming).unwrap();
        let v3_out0 = g.port(v3, 0, portgraph::Direction::Outgoing).unwrap();
        let v3_in1 = g.port(v3, 1, portgraph::Direction::Incoming).unwrap();
        let v3_out1 = g.port(v3, 1, portgraph::Direction::Outgoing).unwrap();
        g.link_ports(v3_out0, v4_in0).unwrap();
        assert_eq!(
            get_line(&g, v2_out0, &mut BTreeSet::new()),
            vec![
                Edge(v2_out0.into(), v3_in1.into()),
                Edge(v3_out1.into(), None)
            ]
        );
    }

    #[test]
    fn get_line_test_2() {
        let mut g = graph();
        let v0 = g.nodes_iter().nth(0).unwrap();
        let v2 = g.nodes_iter().nth(2).unwrap();
        let v3 = g.nodes_iter().nth(3).unwrap();
        let v4 = g.add_node(1, 0);
        let v0_in0 = g.port(v0, 0, portgraph::Direction::Incoming).unwrap();
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_out0 = g.port(v2, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in1 = g.port(v2, 1, portgraph::Direction::Incoming).unwrap();
        let v4_in0 = g.port(v4, 0, portgraph::Direction::Incoming).unwrap();
        // let v4_out0 = g.port(v4, 0, portgraph::Direction::Incoming).unwrap();
        let v3_in1 = g.port(v3, 1, portgraph::Direction::Incoming).unwrap();
        let v3_out1 = g.port(v3, 1, portgraph::Direction::Outgoing).unwrap();
        g.link_ports(v3_out1, v4_in0).unwrap();
        assert_eq!(
            get_line(&g, v2_out0, &mut BTreeSet::new()),
            vec![
                Edge(v2_out0.into(), v3_in1.into()),
                Edge(v3_out1.into(), v4_in0.into())
            ]
        );
        assert_eq!(
            get_line(&g, v2_in1, &mut BTreeSet::new()),
            vec![
                Edge(v2_in1.into(), v0_out0.into()),
                Edge(v0_in0.into(), None)
            ]
        );
    }
}
