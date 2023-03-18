use std::{collections::VecDeque, iter::FusedIterator};

use bitvec::prelude::*;

use portgraph::{NodeIndex, PortGraph};

pub enum Direction {
    Incoming = 0,
    Outgoing = 1,
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
            Direction::Incoming => self.graph.links(next, portgraph::Direction::Incoming),
            Direction::Outgoing => self.graph.links(next, portgraph::Direction::Outgoing),
            Direction::Both => self.graph.all_links(next),
        };
        for neigh in link_ports.filter_map(|p| p.and_then(|p| self.graph.port_node(p))) {
            self.queue.push_back(neigh);
        }
        Some(next)
    }
}

impl<'graph> FusedIterator for PreOrder<'graph> {}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::{Direction, PreOrder};
    use crate::utils::test_utils::*;

    #[test]
    fn preorder() {
        let g = graph();
        let (v0, v1, v2, v3) = g.nodes_iter().collect_tuple().unwrap();

        let it = PreOrder::new(&g, vec![v2], Direction::Outgoing);
        assert_eq!(it.collect::<Vec<_>>(), vec![v2, v3]);

        let it = PreOrder::new(&g, vec![v2], Direction::Incoming);
        assert_eq!(it.collect::<Vec<_>>(), vec![v2, v1, v0]);

        let it = PreOrder::new(&g, vec![v0], Direction::Outgoing);
        assert_eq!(it.collect::<Vec<_>>(), vec![v0, v2, v3]);

        let it = PreOrder::new(&g, vec![v0], Direction::Both);
        assert_eq!(it.collect::<Vec<_>>(), vec![v0, v2, v1, v3]);
    }
}
