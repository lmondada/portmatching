//! Compute skeletons of port graphs.
//!
//! This module contains the [`Skeleton`] data structure, which can be used to
//! compute the spine and ribs of a graph, relative to a root node.
use std::{
    cmp,
    collections::{BTreeMap, VecDeque},
    vec,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex};

use crate::utils::{follow_path, port_opposite, pre_order::shortest_path, ZeroRange};

// use super::{
//     Address, PortGraphAddressing, Rib,
// };
// use crate::constraint::unweighted::{NodeAddress, PortAddress, PortLabel};

use bitvec::prelude::*;

use super::SpineAddress;

type Rib = ZeroRange;

/// A node on a line
#[derive(Clone, Debug)]
pub(super) struct LinePoint {
    pub(super) line_ind: usize,
    pub(super) ind: isize,
    offset: usize,
}

type Spine = Vec<SpineAddress>;

/// Partition graph into paths that form a skeleton
///
/// This data structure can be used to obtain the spine and ribs of a graph,
/// relative to a root node.
#[derive(Debug, Clone)]
pub struct Skeleton<'g> {
    pub(super) node2line: Vec<Vec<LinePoint>>,
    graph: &'g PortGraph,
    pub(super) root: NodeIndex,
    pub(super) spine: Spine,
}

impl<'g> Skeleton<'g> {
    /// A reference to the skeleton's graph
    pub fn graph(&self) -> &PortGraph {
        self.graph
    }

    /// The root node of the graph of the skeleton
    pub fn root(&self) -> NodeIndex {
        self.root
    }

    /// Create a [`Skeleton`] from a graph and a root node.
    pub fn new(graph: &'g PortGraph, root: NodeIndex) -> Self {
        let mut port_queue = VecDeque::from_iter(graph.all_ports(root));
        let mut node2line = vec![Vec::new(); graph.node_capacity()];
        let mut visited_ports = bitvec![0; graph.port_capacity()];
        let mut line_cnt = 0;
        if port_queue.is_empty() {
            // edge case with only root: add an empty (0, 0) line
            node2line[root.index()].push(LinePoint {
                line_ind: 0,
                ind: 0,
                offset: 0,
            });
        }
        while let Some(port) = port_queue.pop_front() {
            if visited_ports[port.index()] {
                continue;
            }
            let line = get_line(graph, port);
            let ind_offset = line
                .ports
                .iter()
                .enumerate()
                .flat_map(|(i, ps)| ps.iter().flatten().map(move |&p| (i, p)))
                .find(|&(_, p)| p == port)
                .expect("line must contain port")
                .0 as isize;
            if line.is_cyclic {
                assert_eq!(ind_offset, 0);
            }
            for (i, ps) in line.ports.iter().enumerate() {
                let &p = ps.iter().flatten().next().expect("must have one port");
                let node = graph.port_node(p).expect("invalid port");
                let offset = graph.port_offset(p).expect("invalid port").index();
                node2line[node.index()].push(LinePoint {
                    line_ind: line_cnt,
                    ind: (i as isize) - ind_offset,
                    offset,
                });
                if line.is_cyclic && i != 0 {
                    node2line[node.index()].push(LinePoint {
                        line_ind: line_cnt,
                        ind: (i as isize) - (line.ports.len() as isize),
                        offset,
                    });
                }
                for &p in ps.iter().flatten() {
                    visited_ports.set(p.index(), true);
                }
                port_queue.extend(graph.all_ports(node));
            }
            line_cnt += 1;
        }
        let spine = Self::compute_spine(&node2line, graph, root);
        Self {
            node2line,
            graph,
            root,
            spine,
        }
    }

    fn compute_spine(node2line: &[Vec<LinePoint>], graph: &PortGraph, root: NodeIndex) -> Spine {
        let mut spine = Vec::new();
        for line_ind in 0.. {
            let nodes_on_line: Vec<_> = node2line
                .iter()
                .enumerate()
                .filter(|(_, lines)| lines.iter().any(|line| line.line_ind == line_ind))
                .map(|(n, _)| NodeIndex::new(n))
                .collect();
            if nodes_on_line.is_empty() {
                break;
            }
            let path =
                shortest_path(graph, [root], nodes_on_line.iter().copied()).expect("got no path");
            let spine_lp = node2line[path.target.index()]
                .iter()
                .filter(|line| line.line_ind == line_ind)
                .min_by_key(|line| line.ind.abs())
                .expect("invalid spine of path");
            spine.push(SpineAddress {
                path: path.out_ports.into(),
                offset: spine_lp.offset,
            });
        }
        spine
    }

    // pub(crate) fn get_spine(&self) -> &Spine {
    //     &self.spine
    // }

    /// The ribs of the graph, relative to the spine
    ///
    /// The spine can be any set of nodes of the graph, even if it does not form
    /// a complete spine of the entire graph
    pub(crate) fn get_ribs(&self, spine: &[SpineAddress]) -> Vec<Rib> {
        // Compute intervals
        // All indices that we must represent must be in the interval
        let spine_len = cmp::max(spine.len(), 1);
        let mut ribs = vec![ZeroRange::Empty; spine_len];
        let all_addrs = self
            .graph
            .nodes_iter()
            .flat_map(|n| self.get_all_addresses(n, spine));
        for (l_ind, ind) in all_addrs {
            ribs[l_ind].insert(ind);
        }
        ribs
    }

    /// All the addresses of `node`, relative to the spine
    fn get_all_addresses(&self, node: NodeIndex, spine: &[SpineAddress]) -> Vec<(usize, isize)> {
        if node == self.root {
            return vec![(0, 0)];
        }
        let spine = self.instantiate_spine(spine);
        let mut rev_inds: BTreeMap<_, Vec<_>> = Default::default();
        for (i, lp) in spine.iter().enumerate() {
            if let Some(lp) = lp {
                rev_inds.entry(lp.line_ind).or_default().push(i);
            }
        }
        let mut all_addrs = Vec::new();
        for line in self.node2line[node.index()].iter() {
            for &spine_ind in rev_inds.get(&line.line_ind).unwrap_or(&Vec::new()) {
                let spine = spine[spine_ind].expect("By construction of in rev_inds");
                let ind = line.ind - spine.ind;
                all_addrs.push((spine_ind, ind))
            }
        }
        all_addrs.sort_unstable();
        all_addrs
    }

    /// Add `addr` to the spine
    ///
    /// A spine that does not cover the entire graph can be extended by adding
    /// new vertebrae until it does. This is achieved in this method by adding
    /// the node addressed by `addr` to the spine.
    ///
    /// Return the new address of the node
    // pub(crate) fn extend_spine(
    //     &self,
    //     spine: &mut Spine,
    //     addr: &Address<usize>,
    //     port: PortOffset,
    // ) -> Address<usize> {
    //     let &(l_ind, ind) = addr;
    //     let path_to_spine = spine[l_ind].0.clone();
    //     let (spine_ind, l_ind) = {
    //         let spine_root = follow_path(&path_to_spine, self.root, self.graph)
    //             .expect("cannot reach spine root");
    //         let line = self.node2line[spine_root.index()]
    //             .iter()
    //             .find(|l| l.offset == spine[l_ind].1)
    //             .expect("Spine root is not root");
    //         (line.ind, line.line_ind)
    //     };
    //     let mut spine_to_node = vec![None; ind.unsigned_abs()];
    //     self.node2line
    //         .iter()
    //         .enumerate()
    //         .flat_map(|(n, lines)| {
    //             let node = NodeIndex::new(n);
    //             lines.iter().map(move |line| (line, node))
    //         })
    //         .filter(|(line, _)| {
    //             let spine_dst = line.ind - spine_ind;
    //             line.line_ind == l_ind && ind * spine_dst >= 0 && spine_dst.abs() < ind.abs()
    //         })
    //         .for_each(|(line, node)| {
    //             let spine_dst = line.ind - spine_ind;
    //             // We check the sign of `ind` (instead of `spine_ind`) because
    //             // the sign is always the same, except for the ambiguous
    //             // `spine_ind == 0` case
    //             let port = if ind < 0 {
    //                 self.graph.input(node, line.offset)
    //             } else {
    //                 self.graph.output(node, line.offset)
    //             };
    //             spine_to_node[spine_dst.unsigned_abs()] =
    //                 self.graph.port_offset(port.expect("Cannot follow path"));
    //         });
    //     let mut path = path_to_spine;
    //     path.extend(
    //         spine_to_node
    //             .into_iter()
    //             .map(|ind| ind.expect("Cannot follow path")),
    //     );
    //     spine.push((path, port.index()));
    //     (spine.len() - 1, 0)
    // }

    /// Return the spine as a list of nodes of [`self.graph`]
    pub(super) fn instantiate_spine(&self, spine: &[SpineAddress]) -> Vec<Option<&LinePoint>> {
        spine
            .iter()
            .map(|SpineAddress { path, offset }| {
                let n = follow_path(path, self.root, self.graph)?;
                self.node2line[n.index()]
                    .iter()
                    .find(|line| line.offset == *offset)
            })
            .collect()
    }
}

#[derive(Debug)]
struct Line {
    ports: Vec<[Option<PortIndex>; 2]>,
    is_cyclic: bool,
}

fn get_line(graph: &PortGraph, port: PortIndex) -> Line {
    let (in_port, out_port) = match graph.port_direction(port).expect("invalid port") {
        Direction::Incoming => (Some(port), port_opposite(port, graph)),
        Direction::Outgoing => (port_opposite(port, graph), Some(port)),
    };
    // Start is always an out_port
    let mut start = out_port;
    let mut some_prev = in_port.and_then(|p| graph.port_link(p));
    let mut is_cyclic = false;
    while let Some(prev) = some_prev {
        start = some_prev;
        if start == out_port {
            is_cyclic = true;
            break;
        }
        some_prev = port_opposite(prev, graph).and_then(|p| graph.port_link(p));
    }

    let mut line = Vec::new();
    if let Some(start) = start {
        line.push([port_opposite(start, graph), Some(start)]);
    }

    let mut some_curr = if let Some(start) = start {
        graph.port_link(start)
    } else {
        in_port
    };
    while let Some(curr) = some_curr {
        let curr_op = port_opposite(curr, graph);
        if curr_op.is_some() && curr_op == start {
            assert!(is_cyclic);
            break;
        }
        line.push([some_curr, curr_op]);
        some_curr = curr_op.and_then(|p| graph.port_link(p));
    }

    Line {
        ports: line,
        is_cyclic,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;
    use smallvec::{smallvec, SmallVec};

    use super::Skeleton;
    use crate::constraint::{NodeAddress, SpineAddress};
    use crate::utils::test_utils::gen_portgraph_connected;

    use portgraph::{NodeIndex, PortGraph, PortOffset};

    use portgraph::proptest::gen_node_index;

    proptest! {
        #[test]
        fn prop_line_partition(g in gen_portgraph_connected(10, 4, 20)) {
            let root = NodeIndex::new(0);
            let addressing = Skeleton::new(&g, root).node2line;

            // Every node has an entry (this is a capacity -- not exact)
            prop_assert!(addressing.len() >= g.node_count());

            // Every port appears exactly once
            let mut port_cnt = BTreeMap::new();
            for (n, v) in addressing.iter().enumerate().flat_map(|(n, p)| {
                let n = NodeIndex::new(n);
                p.iter().map(move |l| (n, l))
            }) {
                if let Some(in_port) = g.input(n, v.offset) {
                    *port_cnt.entry(in_port).or_insert(0) += 1;
                }
                if let Some(out_port) = g.output(n, v.offset) {
                    *port_cnt.entry(out_port).or_insert(0) += 1;
                }
            }
            prop_assert_eq!(port_cnt.len(), g.port_count());
            // At most two entries for each port
            prop_assert!(port_cnt.values().all(|&v| v <= 2));
        }
    }

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

    // #[test]
    // fn a_simple_addr() {
    //     let mut g = PortGraph::new();
    //     g.add_node(2, 0);
    //     g.add_node(0, 2);
    //     link(&mut g, (1, 0), (0, 1));
    //     link(&mut g, (1, 1), (0, 0));
    //     let b = PortGraphAddressing::new(NodeIndex::new(0), &g, None, None);
    //     let spine = vec![(Vec::new(), 0), (Vec::new(), 1)];
    //     let ribs = vec![[-1, 0], [0, 0]];
    //     let b = b.with_spine(&spine).with_ribs(&ribs);
    //     let addr = b.get_addr(NodeIndex::new(1), &mut ()).unwrap();
    //     let root = (&[] as &[PortOffset], 0);
    //     assert_eq!(addr, (root, -1));
    // }

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

        let skel = Skeleton::new(&g, NodeIndex::new(0));

        let root = SpineAddress {
            path: SmallVec::new(),
            offset: 0,
        };
        assert_eq!(
            skel.get_node_addr(n0),
            NodeAddress {
                spine: root.clone(),
                ind: 0
            }
        );
        assert_eq!(
            skel.get_node_addr(n2),
            NodeAddress {
                spine: root.clone(),
                ind: 2
            }
        );
        assert_eq!(
            skel.get_node_addr(n4),
            NodeAddress {
                spine: root,
                ind: -1
            }
        );
        let spine = SpineAddress {
            path: smallvec![PortOffset::new_incoming(0)],
            offset: 1,
        };
        assert_eq!(
            skel.get_node_addr(n5),
            NodeAddress {
                spine: spine.clone(),
                ind: 1
            }
        );
        assert_eq!(skel.get_node_addr(n6), NodeAddress { spine, ind: -1 });
    }

    // #[test]
    // fn test_get_addr_cylic() {
    //     let mut g = PortGraph::new();
    //     g.add_node(1, 1);
    //     let n1 = g.add_node(1, 1);
    //     let n2 = g.add_node(1, 1);
    //     link(&mut g, (0, 0), (1, 0));
    //     link(&mut g, (1, 0), (2, 0));
    //     link(&mut g, (2, 0), (0, 0));

    //     let skel = Skeleton::new(&g, NodeIndex::new(0));
    //     let addressing = PortGraphAddressing::from_skeleton(&skel);

    //     let ribs = vec![[0, 2]];
    //     let addressing = addressing.with_ribs(&ribs);
    //     let root_addr = (&[] as &[PortOffset], 0);
    //     assert_eq!(addressing.get_addr(n2, &mut ()).unwrap(), (root_addr, 2));

    //     let ribs = vec![[-2, 0]];
    //     let addressing = addressing.with_ribs(&ribs);
    //     assert_eq!(addressing.get_addr(n2, &mut ()).unwrap(), (root_addr, -1));
    //     assert_eq!(addressing.get_addr(n1, &mut ()).unwrap(), (root_addr, -2));
    // }

    proptest! {
        #[test]
        fn prop_get_addr((g, n) in gen_node_index(gen_portgraph_connected(10, 4, 20))) {
            let skel = Skeleton::new(&g, NodeIndex::new(0));
            let addr = skel.get_node_addr(n);
            prop_assert_eq!(n, addr.get_node(&g, NodeIndex::new(0)).unwrap());
        }
    }
}
