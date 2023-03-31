use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

use super::pre_order::{shortest_path, Path};

#[derive(Debug)]
struct LinePartition<'graph> {
    node2line: BTreeMap<NodeIndex, Vec<LinePoint>>,
    graph: &'graph PortGraph,
    root: NodeIndex,
}

impl<'graph> LinePartition<'graph> {
    pub(crate) fn new(graph: &'graph PortGraph, root: NodeIndex) -> Self {
        let mut port_queue =
            BinaryHeap::from_iter(graph.all_ports(root).map(|p| (Reverse(p), None)));
        let mut node2line: BTreeMap<_, Vec<_>> = Default::default();
        let mut visited_ports = BTreeSet::new();
        let mut line_cnt = 0;
        while let Some((Reverse(port), line_ind)) = port_queue.pop() {
            let port_op = port_opposite(port, graph);
            if !visited_ports.insert(port) {
                continue;
            }
            visited_ports.extend(port_op);
            let node = graph.port_node(port).expect("invalid port");
            let line_ind = line_ind.map(|Reverse(u)| u).unwrap_or_else(|| {
                line_cnt += 1;
                line_cnt - 1
            });
            node2line
                .entry(node)
                .or_default()
                .push(LinePoint::new(graph, line_ind, port, port_op));
            let ports: Vec<_> = [port].into_iter().chain(port_op).collect();
            for port in ports.iter().filter_map(|&p| graph.port_link(p)) {
                if !visited_ports.contains(&port) {
                    port_queue.push((Reverse(port), Some(Reverse(line_ind))));
                }
            }
            for port in graph
                .all_ports(node)
                .filter(|p| !ports.contains(p) && !visited_ports.contains(p))
            {
                port_queue.push((Reverse(port), None))
            }
        }
        Self {
            node2line,
            graph,
            root,
        }
    }

    pub(crate) fn get_skeleton(&self) -> Skeleton {
        let mut skeleton_paths = Vec::new();
        for line_ind in 0.. {
            let nodes_on_line: Vec<_> = self
                .node2line
                .iter()
                .filter(|(_, lines)| lines.iter().any(|line| line.line_ind == line_ind))
                .map(|(&n, _)| n)
                .collect();
            if nodes_on_line.is_empty() {
                break;
            }
            let path = shortest_path(self.graph, [self.root], nodes_on_line).expect("got no path");
            skeleton_paths.push(path);
        }
        skeleton_paths
    }

    pub(crate) fn get_address(&self, skeleton: &Skeleton, graph: &PortGraph) -> (usize, usize) {
        todo!()
    }
}

type Skeleton = Vec<Path>;

fn port_opposite(port: PortIndex, graph: &PortGraph) -> Option<PortIndex> {
    let node = graph.port_node(port).expect("invalid port");
    let offset = graph.port_offset(port).expect("invalid port");
    let offset_op = match offset.direction() {
        Direction::Incoming => PortOffset::new_outgoing(offset.index()),
        Direction::Outgoing => PortOffset::new_incoming(offset.index()),
    };
    graph.port_index(node, offset_op)
}

/// A node on a line
#[derive(Debug)]
struct LinePoint {
    line_ind: usize,
    in_port: Option<PortIndex>,
    out_port: Option<PortIndex>,
}

impl LinePoint {
    fn new(
        graph: &PortGraph,
        line_ind: usize,
        port: PortIndex,
        port_op: Option<PortIndex>,
    ) -> Self {
        // A slightly verbose but straight-forward way of doing this
        match (
            graph.port_direction(port),
            port_op.and_then(|port_op| graph.port_direction(port_op)),
        ) {
            (None, None)
            | (Some(Direction::Outgoing), Some(Direction::Outgoing))
            | (Some(Direction::Incoming), Some(Direction::Incoming)) => {
                panic!("invalid ports for line point")
            }
            (None, Some(Direction::Incoming)) => Self {
                line_ind,
                in_port: port_op,
                out_port: None,
            },
            (None, Some(Direction::Outgoing)) => Self {
                line_ind,
                in_port: None,
                out_port: port_op,
            },
            (Some(Direction::Incoming), None) => Self {
                line_ind,
                in_port: Some(port),
                out_port: None,
            },
            (Some(Direction::Outgoing), None) => Self {
                line_ind,
                in_port: None,
                out_port: Some(port),
            },
            (Some(Direction::Incoming), Some(Direction::Outgoing)) => Self {
                line_ind,
                in_port: Some(port),
                out_port: port_op,
            },
            (Some(Direction::Outgoing), Some(Direction::Incoming)) => Self {
                line_ind,
                in_port: port_op,
                out_port: Some(port),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use portgraph::{NodeIndex, PortGraph, PortOffset};
    use proptest::prelude::*;

    use crate::utils::{skeleton_address::LinePartition, test_utils::gen_portgraph_connected};

    fn link(graph: &mut PortGraph, (out_n, out_p): (usize, usize), (in_n, in_p): (usize, usize)) {
        let out_n = NodeIndex::new(out_n);
        let in_n = NodeIndex::new(in_n);
        let out_p = graph
            .port_index(out_n, PortOffset::new_outgoing(out_p))
            .unwrap();
        let in_p = graph
            .port_index(in_n, PortOffset::new_incoming(in_p))
            .unwrap();
        graph.link_ports(out_p, in_p).unwrap();
    }

    #[test]
    fn test_create_line_partition() {
        let mut g = PortGraph::new();
        g.add_node(2, 3);
        g.add_node(1, 2);
        g.add_node(2, 1);
        link(&mut g, (0, 0), (1, 0));
        link(&mut g, (1, 0), (2, 0));

        link(&mut g, (1, 1), (2, 1));

        g.add_node(1, 0);
        link(&mut g, (0, 1), (3, 0));

        g.add_node(2, 1);
        link(&mut g, (4, 0), (0, 0));

        let partition = LinePartition::new(&g, NodeIndex::new(0)).node2line;
        assert_eq!(partition[&NodeIndex::new(0)].len(), 3);
        assert_eq!(partition[&NodeIndex::new(1)].len(), 2);
        assert_eq!(partition[&NodeIndex::new(2)].len(), 2);
        assert_eq!(partition[&NodeIndex::new(3)].len(), 1);
        assert_eq!(partition[&NodeIndex::new(4)].len(), 2);
    }

    proptest! {
        #[test]
        fn line_partition(g in gen_portgraph_connected(10, 4, 20)) {
            let root = NodeIndex::new(0);
            let partition = LinePartition::new(&g, root).node2line;

            // Every non-trivial node has an entry
            let n_nodes_with_ports = g
                .nodes_iter()
                .filter(|&n| g.num_inputs(n) > 0 || g.num_outputs(n) > 0)
                .count();
            prop_assert_eq!(partition.len(), n_nodes_with_ports);

            // Every port appears exactly once
            let mut port_set = BTreeSet::new();
            for v in partition.values().flatten() {
                if let Some(in_port) = v.in_port {
                    prop_assert!(port_set.insert(in_port));
                }
                if let Some(out_port) = v.out_port {
                    prop_assert!(port_set.insert(out_port));
                }
            }
            prop_assert_eq!(port_set.len(), g.port_count());
        }
    }
}
