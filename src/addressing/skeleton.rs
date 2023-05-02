//! Compute skeletons of port graphs.
//!
//! This module contains the [`Skeleton`] data structure, which can be used to
//! compute the spine and ribs of a graph, relative to a root node.
use std::{
    cmp,
    collections::{BTreeMap, VecDeque},
    vec,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::utils::{follow_path, port_opposite, pre_order::shortest_path};

use super::{Address, PortGraphAddressing, Rib};

type Spine = super::Spine<(Vec<PortOffset>, usize)>;
type SpineRef<'a> = &'a [(Vec<PortOffset>, usize)];

use bitvec::prelude::*;

/// A node on a line
#[derive(Clone, Debug)]
struct LinePoint {
    line_ind: usize,
    ind: isize,
    offset: usize,
}

/// Partition graph into paths that form a skeleton
///
/// This data structure can be used to obtain the spine and ribs of a graph,
/// relative to a root node.
#[derive(Debug)]
pub struct Skeleton<'g> {
    node2line: Vec<Vec<LinePoint>>,
    graph: &'g PortGraph,
    root: NodeIndex,
    spine: Spine,
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
            spine.push((path.out_ports.clone(), spine_lp.offset));
        }
        spine
    }

    pub(crate) fn get_spine(&self) -> &Spine {
        &self.spine
    }

    /// The ribs of the graph, relative to the spine
    ///
    /// The spine can be any set of nodes of the graph, even if it does not form
    /// a complete spine of the entire graph
    pub(crate) fn get_ribs(&self, spine: SpineRef<'_>) -> Vec<Rib> {
        // Compute intervals
        // All indices that we must represent must be in the interval
        let spine_len = cmp::max(spine.len(), 1);
        let mut ribs = vec![[0, -1]; spine_len];
        let all_addrs = self
            .graph
            .nodes_iter()
            .flat_map(|n| self.get_all_addresses(n, spine));
        for (l_ind, ind) in all_addrs {
            let [min, max] = &mut ribs[l_ind];
            *min = cmp::min(*min, ind);
            *max = cmp::max(*max, ind);
        }
        ribs
    }

    /// All the addresses of `node`, relative to the spine
    fn get_all_addresses(&self, node: NodeIndex, spine: SpineRef<'_>) -> Vec<Address<usize>> {
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
    pub(crate) fn extend_spine(
        &self,
        spine: &mut Spine,
        addr: &Address<usize>,
        port: PortOffset,
    ) -> Address<usize> {
        let &(l_ind, ind) = addr;
        let path_to_spine = spine[l_ind].0.clone();
        let (spine_ind, l_ind) = {
            let spine_root = follow_path(&path_to_spine, self.root, self.graph)
                .expect("cannot reach spine root");
            let line = self.node2line[spine_root.index()]
                .iter()
                .find(|l| l.offset == spine[l_ind].1)
                .expect("Spine root is not root");
            (line.ind, line.line_ind)
        };
        let mut spine_to_node = vec![None; ind.unsigned_abs()];
        self.node2line
            .iter()
            .enumerate()
            .flat_map(|(n, lines)| {
                let node = NodeIndex::new(n);
                lines.iter().map(move |line| (line, node))
            })
            .filter(|(line, _)| {
                let spine_dst = line.ind - spine_ind;
                line.line_ind == l_ind && ind * spine_dst >= 0 && spine_dst.abs() < ind.abs()
            })
            .for_each(|(line, node)| {
                let spine_dst = line.ind - spine_ind;
                // We check the sign of `ind` (instead of `spine_ind`) because
                // the sign is always the same, except for the ambiguous
                // `spine_ind == 0` case
                let port = if ind < 0 {
                    self.graph.input(node, line.offset)
                } else {
                    self.graph.output(node, line.offset)
                };
                spine_to_node[spine_dst.unsigned_abs()] =
                    self.graph.port_offset(port.expect("Cannot follow path"));
            });
        let mut path = path_to_spine;
        path.extend(
            spine_to_node
                .into_iter()
                .map(|ind| ind.expect("Cannot follow path")),
        );
        spine.push((path, port.index()));
        (spine.len() - 1, 0)
    }

    /// Return the spine as a list of nodes of [`self.graph`]
    fn instantiate_spine(&self, spine: SpineRef<'_>) -> Vec<Option<&LinePoint>> {
        spine
            .iter()
            .map(|(path, out_port)| {
                let n = follow_path(path, self.root, self.graph)?;
                self.node2line[n.index()]
                    .iter()
                    .find(|line| line.offset == *out_port)
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

impl<'g: 'n, 'n> PortGraphAddressing<'g, 'n, (Vec<PortOffset>, usize)> {
    /// Create a new [`PortGraphAddressing`] from a [`Skeleton`]
    pub fn from_skeleton(skel: &'g Skeleton) -> Self {
        let spine = &skel.spine;
        // let ribs = skel.get_ribs(spine);
        PortGraphAddressing::new(
            skel.root,
            skel.graph,
            Some(spine),
            None, // Some(&ribs)
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use portgraph::NodeIndex;
    use proptest::prelude::*;

    use super::Skeleton;
    use crate::utils::test_utils::gen_portgraph_connected;

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
}
