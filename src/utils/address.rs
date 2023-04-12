use std::{
    cmp,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

use super::pre_order::shortest_path;

pub(crate) struct LinePartition<'graph> {
    pub(crate) node2line: BTreeMap<NodeIndex, Vec<LinePoint>>,
    pub(crate) graph: &'graph PortGraph,
    pub(crate) root: NodeIndex,
}

impl<'graph> fmt::Debug for LinePartition<'graph> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}\n{:#?}", self.node2line, self.root)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Skeleton {
    pub(crate) spine: Spine,
    pub(crate) ribs: Ribs,
}

// impl Skeleton {
//     pub(crate) fn ribs_for(&self, partition: &LinePartition) -> Ribs {
//         let mut ribs = vec![[None; 2]; self.spine.len()];
//         for n in partition.graph.nodes_iter() {
//             if let Some((a, b)) = partition.get_address(n, self, true) {
//                 let [min, max] = &mut ribs[a];
//                 let min = min.get_or_insert(b);
//                 let max = max.get_or_insert(b);
//                 *min = cmp::min(*min, b);
//                 *max = cmp::max(*max, b);
//             }
//         }
//         Ribs(
//             ribs.into_iter()
//                 .map(|[a, b]| [a.unwrap_or(0), b.unwrap_or(0)])
//                 .collect(),
//         )
//     }
// }

#[derive(PartialEq, Eq, Clone, Debug)]
pub(crate) struct Address(pub(crate) usize, pub(crate) i32);

impl Address {
    fn key(&self) -> (usize, u32, bool) {
        let &Address(fst, snd) = self;
        (fst, snd.abs() as u32, snd < 0)
    }
}

impl PartialOrd for Address {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.key().partial_cmp(&other.key())
    }
}

impl Ord for Address {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

pub(crate) type Spine = Vec<(Vec<PortOffset>, usize)>;
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Ribs(pub(crate) Vec<[i32; 2]>);

impl Ribs {
    pub(crate) fn within(&self, ribs: &Ribs) -> bool {
        self.0
            .iter()
            .zip(ribs.0.iter())
            .all(|(a, b)| a[0] >= b[0] && a[1] <= b[1])
    }

    pub(crate) fn add_addr(&mut self, Address(line, ind): Address) {
        let [min, max] = &mut self.0[line];
        *min = cmp::min(*min, ind);
        *max = cmp::max(*max, ind);
    }
}

impl<'graph> LinePartition<'graph> {
    pub(crate) fn new(graph: &'graph PortGraph, root: NodeIndex) -> Self {
        let mut port_queue = VecDeque::from_iter(graph.all_ports(root));
        let mut node2line: BTreeMap<_, _> = graph.nodes_iter().map(|n| (n, Vec::new())).collect();
        let mut visited_ports: BTreeSet<PortIndex> = BTreeSet::new();
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
        while let Some(port) = port_queue.pop_front() {
            if visited_ports.contains(&port) {
                continue;
            }
            let line = get_line(graph, port);
            let ind_offset = line
                .ports
                .iter()
                .enumerate()
                .flat_map(|(i, ps)| ps.into_iter().flatten().map(move |&p| (i, p)))
                .find(|&(_, p)| p == port)
                .expect("line must contain port")
                .0 as i32;
            if line.is_cyclic {
                assert_eq!(ind_offset, 0);
            }
            for (i, ps) in line.ports.iter().enumerate() {
                let &p = ps.iter().flatten().next().expect("must have one port");
                let node = graph.port_node(p).expect("invalid port");
                node2line
                    .get_mut(&node)
                    .expect("added all nodes")
                    .push(LinePoint {
                        line_ind: line_cnt,
                        ind: (i as i32) - ind_offset,
                        in_port: ps[0],
                        out_port: ps[1],
                    });
                if line.is_cyclic && i != 0 {
                    node2line
                        .get_mut(&node)
                        .expect("added all nodes")
                        .push(LinePoint {
                            line_ind: line_cnt,
                            ind: (i as i32) - (line.ports.len() as i32),
                            in_port: ps[0],
                            out_port: ps[1],
                        });
                }
                visited_ports.extend(ps.iter().flatten());
                port_queue.extend(graph.all_ports(node));
            }
            line_cnt += 1;
        }
        Self {
            node2line,
            graph,
            root,
        }
    }

    pub(crate) fn get_skeleton(&self) -> Skeleton {
        let mut spine = Vec::new();
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
            let path = shortest_path(self.graph, [self.root], nodes_on_line.iter().copied())
                .expect("got no path");
            let spine_lp = self.node2line[&path.target]
                .iter()
                .filter(|line| line.line_ind == line_ind)
                .min_by_key(|line| line.ind.abs())
                .expect("invalid spine of path");
            if let Some(port) = spine_lp.out_port.or(spine_lp.in_port) {
                spine.push((
                    path.out_ports.clone(),
                    self.graph.port_offset(port).expect("invalid port").index(),
                ));
            }
        }

        let ribs = self.get_ribs(&spine);
        Skeleton { spine, ribs }
    }

    /// Find the address of `node` relative to the skeleton
    pub(crate) fn get_all_addresses(&self, node: NodeIndex, spine: &Spine) -> Vec<Address> {
        if node == self.root {
            return vec![Address(0, 0)];
        }
        let spine = self.get_spine(spine);
        let mut rev_inds: BTreeMap<_, Vec<_>> = Default::default();
        for (i, lp) in spine.iter().enumerate() {
            if let Some(lp) = lp {
                rev_inds.entry(lp.line_ind).or_default().push(i);
            }
        }
        let mut all_addrs = Vec::new();
        for line in self.node2line[&node].iter() {
            for &spine_ind in rev_inds.get(&line.line_ind).unwrap_or(&Vec::new()) {
                let spine = spine[spine_ind].expect("By construction of in rev_inds");
                let ind = line.ind - spine.ind;
                all_addrs.push(Address(spine_ind, ind))
            }
        }
        all_addrs.sort_unstable();
        all_addrs
    }

    /// Find the address of `node` relative to the skeleton
    pub(crate) fn get_address(
        &self,
        node: NodeIndex,
        spine: &Spine,
        ribs: Option<&Ribs>,
    ) -> Option<Address> {
        let all_addrs = self.get_all_addresses(node, spine);
        let Some(Ribs(ribs)) = ribs else {
            return all_addrs.into_iter().next()
        };
        all_addrs
            .into_iter()
            .filter(|&Address(spine_ind, ind)| {
                let [from, to] = ribs[spine_ind];
                from <= ind && to >= ind
            })
            .next()
    }

    pub(crate) fn get_node_index(
        &self,
        &Address(line_ind, ind): &Address,
        spine: &Spine,
    ) -> Option<NodeIndex> {
        if (line_ind, ind) == (0, 0) {
            return Some(self.root);
        }
        let skeleton_lines = self.get_spine(spine);
        let line = skeleton_lines[line_ind]?;
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

    fn get_spine(&self, spine: &Spine) -> Vec<Option<&LinePoint>> {
        spine
            .iter()
            .map(|(path, out_port)| {
                let n = follow_path(path, self.root, self.graph)?;
                self.node2line[&n].iter().find(|line| {
                    for port in [line.out_port, line.in_port] {
                        if let Some(port) = port {
                            let offset = self.graph.port_offset(port).expect("invalid port");
                            if offset.index() == *out_port {
                                return true;
                            }
                        }
                    }
                    return false;
                })
            })
            .collect()
    }

    pub(crate) fn get_ribs(&self, spine: &Spine) -> Ribs {
        // Compute intervals
        // All indices that we must represent must be in the interval
        let spine_len = cmp::max(spine.len(), 1);
        let mut ribs = vec![[0, 0]; spine_len];
        let all_addrs = self
            .graph
            .nodes_iter()
            .flat_map(|n| self.get_all_addresses(n, spine));
        for Address(l_ind, ind) in all_addrs {
            let [min, max] = &mut ribs[l_ind];
            *min = cmp::min(*min, ind);
            *max = cmp::max(*max, ind);
        }
        Ribs(ribs)
    }

    pub(crate) fn extend_spine(
        &self,
        spine: &mut Spine,
        &Address(l_ind, ind): &Address,
        port: PortOffset,
    ) -> Address {
        let path_to_spine = spine[l_ind].0.clone();
        let (spine_ind, l_ind) = {
            let spine_root = follow_path(&path_to_spine, self.root, self.graph)
                .expect("cannot reach spine root");
            let line = self.node2line[&spine_root]
                .iter()
                .find(|l| {
                    let Some(port) = l.out_port.or(l.in_port) else {
                        return false
                    };
                    let offset = self.graph.port_offset(port).expect("invalid port");
                    offset.index() == spine[l_ind].1
                })
                .expect("Spine root is not root");
            (line.ind, line.line_ind)
        };
        let mut spine_to_node = vec![None; ind.abs() as usize];
        self.node2line
            .values()
            .flatten()
            .filter(|line| {
                let spine_dst = line.ind - spine_ind;
                line.line_ind == l_ind && ind * spine_dst >= 0 && spine_dst.abs() < ind.abs()
            })
            .for_each(|line| {
                let spine_dst = line.ind - spine_ind;
                let port = if ind < 0 { line.in_port } else { line.out_port };
                spine_to_node[spine_dst.abs() as usize] =
                    self.graph.port_offset(port.expect("Cannot follow path"));
            });
        let mut path = path_to_spine;
        path.extend(
            spine_to_node
                .into_iter()
                .map(|ind| ind.expect("Cannot follow path")),
        );
        spine.push((path, port.index()));
        Address(spine.len() - 1, 0)
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

fn follow_path(path: &Vec<PortOffset>, root: NodeIndex, graph: &PortGraph) -> Option<NodeIndex> {
    let mut curr_node = root;
    for &offset in path {
        let out_port = graph.port_index(curr_node, offset)?;
        let in_port = graph.port_link(out_port)?;
        curr_node = graph.port_node(in_port)?;
    }
    Some(curr_node)
}

pub(crate) fn port_opposite(port: PortIndex, graph: &PortGraph) -> Option<PortIndex> {
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
pub(crate) struct LinePoint {
    line_ind: usize,
    ind: i32,
    in_port: Option<PortIndex>,
    out_port: Option<PortIndex>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use portgraph::{NodeIndex, PortGraph, PortOffset};
    use proptest::prelude::*;

    use crate::utils::{
        address::{Address, LinePartition, Ribs, Skeleton},
        test_utils::gen_portgraph_connected,
    };
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
    fn a_simple_addr() {
        let mut g = PortGraph::new();
        g.add_node(2, 0);
        g.add_node(0, 2);
        link(&mut g, (1, 0), (0, 1));
        link(&mut g, (1, 1), (0, 0));
        let p = LinePartition::new(&g, NodeIndex::new(0));
        let skel = Skeleton {
            spine: vec![([].into(), 0), ([].into(), 1)],
            ribs: Ribs(vec![[-1, 0], [0, 0]]),
        };
        let addr = p
            .get_address(NodeIndex::new(1), &skel.spine, Some(&skel.ribs))
            .unwrap();
        assert_eq!(addr, Address(0, -1));
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
        assert_eq!(
            partition
                .get_address(n0, &skeleton.spine, Some(&skeleton.ribs))
                .unwrap(),
            Address(0, 0)
        );
        assert_eq!(
            partition
                .get_address(n2, &skeleton.spine, Some(&skeleton.ribs))
                .unwrap(),
            Address(0, 2)
        );
        assert_eq!(
            partition
                .get_address(n4, &skeleton.spine, Some(&skeleton.ribs))
                .unwrap(),
            Address(0, -1)
        );
        assert_eq!(
            partition
                .get_address(n5, &skeleton.spine, Some(&skeleton.ribs))
                .unwrap(),
            Address(3, 1)
        );
        assert_eq!(
            partition
                .get_address(n6, &skeleton.spine, Some(&skeleton.ribs))
                .unwrap(),
            Address(3, -1)
        );
    }

    #[test]
    fn test_get_addr_cylic() {
        let mut g = PortGraph::new();
        g.add_node(1, 1);
        let n1 = g.add_node(1, 1);
        let n2 = g.add_node(1, 1);
        link(&mut g, (0, 0), (1, 0));
        link(&mut g, (1, 0), (2, 0));
        link(&mut g, (2, 0), (0, 0));

        let partition = LinePartition::new(&g, NodeIndex::new(0));

        let skel = Skeleton {
            spine: vec![([].into(), 0)],
            ribs: Ribs(vec![[0, 2]]),
        };
        assert_eq!(
            partition
                .get_address(n2, &skel.spine, Some(&skel.ribs))
                .unwrap(),
            Address(0, 2)
        );

        let skel = Skeleton {
            spine: vec![([].into(), 0)],
            ribs: Ribs(vec![[-2, 0]]),
        };
        assert_eq!(
            partition
                .get_address(n2, &skel.spine, Some(&skel.ribs))
                .unwrap(),
            Address(0, -1)
        );
        assert_eq!(
            partition
                .get_address(n1, &skel.spine, Some(&skel.ribs))
                .unwrap(),
            Address(0, -2)
        );
    }

    proptest! {
        #[test]
        fn prop_line_partition(g in gen_portgraph_connected(10, 4, 20)) {
            let root = NodeIndex::new(0);
            let partition = LinePartition::new(&g, root).node2line;

            // Every node has an entry
            prop_assert_eq!(partition.len(), g.node_count());

            // Every port appears exactly once
            let mut port_cnt = BTreeMap::new();
            for v in partition.values().flatten() {
                if let Some(in_port) = v.in_port {
                    *port_cnt.entry(in_port).or_insert(0) += 1;
                }
                if let Some(out_port) = v.out_port {
                    *port_cnt.entry(out_port).or_insert(0) += 1;
                }
            }
            prop_assert_eq!(port_cnt.len(), g.port_count());
            // At most two entries for each port
            prop_assert!(port_cnt.values().all(|&v| v <= 2));
        }
    }

    proptest! {
        #[test]
        fn prop_get_addr((g, n) in gen_node_index(gen_portgraph_connected(10, 4, 20))) {
            let p = LinePartition::new(&g, NodeIndex::new(0));
            let skeleton = p.get_skeleton();
            let addr = p.get_address(n, &skeleton.spine, Some(&skeleton.ribs)).unwrap();
            prop_assert_eq!(n, p.get_node_index(&addr, &skeleton.spine).unwrap());
        }
    }
}