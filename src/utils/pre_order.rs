use std::{collections::VecDeque, iter::FusedIterator};

use bitvec::prelude::*;

use portgraph::{LinkView, NodeIndex, PortGraph, PortView};

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
        self.queue.extend(match self.direction {
            Direction::_Incoming => self.graph.neighbours(next, portgraph::Direction::Incoming),
            Direction::_Outgoing => self.graph.neighbours(next, portgraph::Direction::Outgoing),
            Direction::Both => self.graph.all_neighbours(next),
        });
        Some(next)
    }
}

impl<'graph> FusedIterator for PreOrder<'graph> {}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::PortView;

    use super::{Direction, PreOrder};
    use crate::utils::test::*;

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
