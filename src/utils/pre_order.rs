use std::{
    collections::{BTreeSet, VecDeque},
    iter::FusedIterator,
};

use bitvec::prelude::*;

use portgraph::{NodeIndex, PortGraph, PortIndex};

pub enum Direction {
    _Incoming = 0,
    _Outgoing = 1,
    Both = 2,
}

/// Iterator over an `UnweightedGraph` in pre-order.
pub struct PreOrder<'graph> {
    graph: &'graph PortGraph,
    queue: VecDeque<NodeIndex>,
    visited: BitVec,
    direction: Direction,
}

impl<'graph> PreOrder<'graph> {
    pub fn new(
        graph: &'graph PortGraph,
        source: impl IntoIterator<Item = NodeIndex>,
        direction: Direction,
    ) -> Self {
        let visited = bitvec![0; graph.node_capacity()];

        Self {
            graph,
            queue: source.into_iter().collect(),
            visited,
            direction,
        }
    }
}

impl<'graph> Iterator for PreOrder<'graph> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next = self.queue.pop_front()?;
        while self.visited.replace(next.index(), true) {
            next = self.queue.pop_front()?;
        }
        let link_ports = match self.direction {
            Direction::_Incoming => self.graph.links(next, portgraph::Direction::Incoming),
            Direction::_Outgoing => self.graph.links(next, portgraph::Direction::Outgoing),
            Direction::Both => self.graph.all_links(next),
        };
        for neigh in link_ports.filter_map(|p| p.and_then(|p| self.graph.port_node(p))) {
            self.queue.push_back(neigh);
        }
        Some(next)
    }
}

impl<'graph> FusedIterator for PreOrder<'graph> {}

pub fn pre_order(
    graph: &PortGraph,
    source: impl IntoIterator<Item = NodeIndex>,
    direction: Direction,
) -> PreOrder {
    PreOrder::new(graph, source, direction)
}

pub struct Path {
    root: NodeIndex,
    out_ports: Vec<PortIndex>,
}

pub fn shortest_path(
    graph: &PortGraph,
    source: impl IntoIterator<Item = NodeIndex>,
    target: impl IntoIterator<Item = NodeIndex>,
) -> Option<Path> {
    let source: BTreeSet<_> = source.into_iter().collect();
    let target: Vec<_> = target.into_iter().collect();

    let mut distance = vec![usize::MAX; graph.node_capacity()];
    let mut prev = vec![None; graph.node_capacity()];

    for n in source.iter() {
        distance[n.index()] = 0;
    }
    let mut nodes = pre_order(graph, source.iter().copied(), Direction::Both);
    while target.iter().all(|n| distance[n.index()] == usize::MAX) {
        let node = nodes.next()?;
        if let Some(best_out_port) = graph.all_links(node).flatten().min_by_key(|&p| {
            let n = graph.port_node(p).expect("invalid port");
            distance[n.index()]
        }) {
            let min = distance[best_out_port.index()];
            if min + 1 < distance[node.index()] {
                distance[node.index()] = min + 1;
                prev[node.index()] = Some(best_out_port);
            }
        }
    }
    let mut node = target.iter().min_by_key(|n| distance[n.index()]).copied()?;
    let mut out_ports = Vec::new();
    while !source.contains(&node) {
        let port = prev[node.index()]?;
        out_ports.push(port);
        node = graph.port_node(port).expect("invalid port");
    }
    Some(Path {
        root: node,
        out_ports: out_ports.into_iter().rev().collect(),
    })
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{Direction, PreOrder};
    use crate::utils::test_utils::*;

    #[test]
    fn preorder() {
        let g = graph();
        let (v0, v1, v2, v3) = g.nodes_iter().collect_tuple().unwrap();

        let it = PreOrder::new(&g, vec![v2], Direction::_Outgoing);
        assert_eq!(it.collect::<Vec<_>>(), vec![v2, v3]);

        let it = PreOrder::new(&g, vec![v2], Direction::_Incoming);
        assert_eq!(it.collect::<Vec<_>>(), vec![v2, v1, v0]);

        let it = PreOrder::new(&g, vec![v0], Direction::_Outgoing);
        assert_eq!(it.collect::<Vec<_>>(), vec![v0, v2, v3]);

        let it = PreOrder::new(&g, vec![v0], Direction::Both);
        assert_eq!(it.collect::<Vec<_>>(), vec![v0, v2, v1, v3]);
    }
}
