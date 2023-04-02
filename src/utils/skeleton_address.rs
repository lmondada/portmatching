use std::{
    cmp::{min_by_key, Reverse},
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
        let mut node2line: BTreeMap<_, _> = graph.nodes_iter().map(|n| (n, Vec::new())).collect();
        let mut visited_ports = BTreeSet::new();
        let mut line_cnt = 0;
        if port_queue.is_empty() {
            // edge case with only root: add an empty (0, 0) line
            node2line
                .get_mut(&root)
                .expect("added all nodes")
                .push(LinePoint {
                    line_ind: 0,
                    ind: 0,
                    in_port: None,
                    out_port: None,
                });
        }
        while let Some((Reverse(port), addr)) = port_queue.pop() {
            let port_op = port_opposite(port, graph);
            if !visited_ports.insert(port) {
                continue;
            }
            visited_ports.extend(port_op);
            let node = graph.port_node(port).expect("invalid port");
            let (line_ind, ind) = addr.map(|(Reverse(u), i)| (u, i)).unwrap_or_else(|| {
                line_cnt += 1;
                (line_cnt - 1, 0)
            });
            node2line
                .get_mut(&node)
                .expect("added all nodes")
                .push(LinePoint::new(graph, line_ind, ind, port, port_op));
            if let Some(port) = port_op.and_then(|p| graph.port_link(p)) {
                if !visited_ports.contains(&port) {
                    let new_ind = if ind >= 0 { ind + 1 } else { ind - 1 };
                    port_queue.push((Reverse(port), Some((Reverse(line_ind), new_ind))));
                }
            }
            if ind == 0 {
                if let Some(port) = graph.port_link(port) {
                    if !visited_ports.contains(&port) {
                        port_queue.push((Reverse(port), Some((Reverse(line_ind), -1))));
                    }
                }
            }
            let ports: Vec<_> = [port].into_iter().chain(port_op.into_iter()).collect();
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
            let lp = self.node2line[&path.target]
                .iter()
                .find(|line| line.line_ind == line_ind)
                .expect("found target above");
            if let Some(out_port) = lp.out_port {
                skeleton_paths.push((
                    path.out_ports.clone(),
                    self.graph.port_offset(out_port).expect("invalid port"),
                ));
            }
            if let Some(in_port) = lp.in_port {
                skeleton_paths.push((
                    path.out_ports,
                    self.graph.port_offset(in_port).expect("invalid port"),
                ));
            }
        }
        skeleton_paths
    }

    /// Find the address of `node` relative to the skeleton
    pub(crate) fn get_address(&self, node: NodeIndex, skeleton: &Skeleton) -> Option<(usize, i32)> {
        if node == self.root {
            return Some((0, 0))
        }
        let skeleton_lines = self.skeleton_lines(skeleton)?;
        let rev_inds: BTreeMap<_, _> = skeleton_lines
            .iter()
            .enumerate()
            .map(|(i, line)| (line.line_ind, i))
            .collect();
        let min_line = self.node2line[&node]
            .iter()
            .min_by_key(|line| rev_inds[&line.line_ind])
            .expect("every node belongs to a line");
        let root = skeleton_lines[rev_inds[&min_line.line_ind]];
        Some((root.line_ind, min_line.ind - root.ind))
    }

    pub(crate) fn get_node_index(
        &self,
        &(line_ind, ind): &(usize, i32),
        skeleton: &Skeleton,
    ) -> Option<NodeIndex> {
        if (line_ind, ind) == (0, 0) {
            return Some(self.root);
        }
        let skeleton_lines = self.skeleton_lines(skeleton)?;
        let line = skeleton_lines[line_ind];
        let mut port = if ind == 0 {
            line.out_port.or(line.in_port)
        } else if ind > 0 {
            line.out_port
        } else {
            line.in_port
        };
        let mut node = self.graph.port_node(port?).expect("invalid port");
        for _ in 0..ind.abs() {
            let next_port = self.graph.port_link(port?)?;
            node = self.graph.port_node(next_port).expect("invalid port");
            port = port_opposite(next_port, self.graph);
        }
        Some(node)
    }

    fn skeleton_lines(&self, skeleton: &Skeleton) -> Option<Vec<&LinePoint>> {
        let skeleton = skeleton
            .into_iter()
            .map(|(path, out_port)| {
                (follow_path(path, self.root, self.graph).map(|n| (n, *out_port)))
            })
            .collect::<Option<Vec<_>>>()?;
        skeleton
            .iter()
            .map(|&(n, p)| {
                self.node2line[&n].iter().find(|line| {
                    for port in [line.out_port, line.in_port] {
                        if let Some(port) = port {
                            let offset = self.graph.port_offset(port).expect("invalid port");
                            if offset == p {
                                return true;
                            }
                        }
                    }
                    return false;
                })
            })
            .collect()
    }
}

fn follow_path(path: &Vec<PortOffset>, root: NodeIndex, graph: &PortGraph) -> Option<NodeIndex> {
    let mut curr_node = root;
    for &offset in path {
        let out_port = graph.port_index(curr_node, offset)?;
        let in_port = graph.port_link(out_port)?;
        curr_node = graph.port_node(in_port)?;
    }
    Some(curr_node)
}

type Skeleton = Vec<(Vec<PortOffset>, PortOffset)>;

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
    ind: i32,
    in_port: Option<PortIndex>,
    out_port: Option<PortIndex>,
}

impl LinePoint {
    fn new(
        graph: &PortGraph,
        line_ind: usize,
        ind: i32,
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
                ind,
                in_port: port_op,
                out_port: None,
            },
            (None, Some(Direction::Outgoing)) => Self {
                line_ind,
                ind,
                in_port: None,
                out_port: port_op,
            },
            (Some(Direction::Incoming), None) => Self {
                line_ind,
                ind,
                in_port: Some(port),
                out_port: None,
            },
            (Some(Direction::Outgoing), None) => Self {
                line_ind,
                ind,
                in_port: None,
                out_port: Some(port),
            },
            (Some(Direction::Incoming), Some(Direction::Outgoing)) => Self {
                line_ind,
                ind,
                in_port: Some(port),
                out_port: port_op,
            },
            (Some(Direction::Outgoing), Some(Direction::Incoming)) => Self {
                line_ind,
                ind,
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
    use portgraph::proptest::gen_node_index;

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

    #[test]
    fn test_get_addr() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 3);
        g.add_node(1, 2);
        let n2 = g.add_node(2, 1);
        link(&mut g, (0, 0), (1, 0));
        link(&mut g, (1, 0), (2, 0));
        link(&mut g, (1, 1), (2, 1));
        g.add_node(1, 0);
        link(&mut g, (0, 1), (3, 0));
        let n4 = g.add_node(2, 2);
        let n5 = g.add_node(1, 0);
        let n6 = g.add_node(0, 1);
        link(&mut g, (4, 0), (0, 0));
        link(&mut g, (4, 1), (5, 0));
        link(&mut g, (6, 0), (4, 1));

        let partition = LinePartition::new(&g, NodeIndex::new(0));

        let skeleton = partition.get_skeleton();
        assert_eq!(partition.get_address(n0, &skeleton).unwrap(), (0, 0));
        assert_eq!(partition.get_address(n2, &skeleton).unwrap(), (0, 2));
        assert_eq!(partition.get_address(n4, &skeleton).unwrap(), (0, -1));
        assert_eq!(partition.get_address(n5, &skeleton).unwrap(), (4, 1));
        assert_eq!(partition.get_address(n6, &skeleton).unwrap(), (4, -1));
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

    proptest! {
        #[test]
        fn get_addr((g, n) in gen_node_index(gen_portgraph_connected(10, 4, 20))) {
            let p = LinePartition::new(&g, NodeIndex::new(0));
            let skeleton = p.get_skeleton();
            dbg!(&skeleton);
            dbg!(g.nodes_iter().collect::<Vec<_>>());
            let addr = p.get_address(n, &skeleton).unwrap();
            dbg!(&addr);
            prop_assert_eq!(n, p.get_node_index(&addr, &skeleton).unwrap());
        }
    }
}
