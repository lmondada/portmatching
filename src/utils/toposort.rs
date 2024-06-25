#![allow(dead_code)] // Temporary until refactor is complete

use std::hash::Hash;

use petgraph::{visit::IntoNeighborsDirected, Direction};

use crate::HashSet;

/// Perform a topological sort of a graph, without reference to it.
///
/// This will only traverse the descendants of `root`.
///
/// Graph mutations of parts of the graph that have not been visited yet are
/// allowed during the traversal.
///
/// If you delete edges or change the graph in other ways, the behaviour
/// is undefined.
pub fn online_toposort<N: Eq + Hash + Copy>(root: N) -> OnlineToposort<N> {
    OnlineToposort::new(root)
}

/// Iterator for online topological traversal of a graph.
pub struct OnlineToposort<N> {
    /// Nodes that have been visited
    visited: HashSet<N>,
    /// Stack of nodes that are ready to be visited (all predecessors have
    /// been visited)
    to_visit_next: Vec<N>,
    /// Nodes that have been visited but whose children have not yet been visited
    ///
    /// We delay processing the outgoing edges of these nodes as long as possible,
    /// to allow for graph edits.
    pending_children: Vec<N>,
}

impl<N: Eq + Hash + Copy> OnlineToposort<N> {
    fn new(root: N) -> Self {
        Self {
            visited: HashSet::default(),
            to_visit_next: vec![root],
            pending_children: vec![],
        }
    }

    pub fn next<G: IntoNeighborsDirected<NodeId = N>>(&mut self, graph: G) -> Option<N> {
        while self.to_visit_next.is_empty() {
            // We must process some pending nodes
            let pending_node = self.pending_children.pop()?;
            for child in graph.neighbors_directed(pending_node, Direction::Outgoing) {
                let mut parents = graph.neighbors_directed(child, Direction::Incoming);
                if parents.all(|p| self.visited.contains(&p)) {
                    // Only insert if it hasn't been visited yet nor is it queued to be
                    if !self.visited.contains(&child) && !self.to_visit_next.contains(&child) {
                        self.to_visit_next.push(child);
                    }
                }
            }
        }
        let next_state = self.to_visit_next.pop()?;
        self.visited.insert(next_state);
        self.pending_children.push(next_state);

        Some(next_state)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::from_fn;

    use itertools::Itertools;
    use petgraph::stable_graph::StableDiGraph;
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn graph() -> StableDiGraph<usize, ()> {
        let mut graph = StableDiGraph::new();
        let nodes = (0..4).map(|i| graph.add_node(i)).collect_vec();
        graph.add_edge(nodes[0], nodes[2], ());
        graph.add_edge(nodes[2], nodes[3], ());
        graph.add_edge(nodes[3], nodes[1], ());
        graph
    }

    #[rstest]
    fn test_online_toposort(graph: StableDiGraph<usize, ()>) {
        let mut toposort = online_toposort(0.into());
        assert_eq!(
            from_fn(|| toposort.next(&graph)).collect_vec(),
            vec![0.into(), 2.into(), 3.into(), 1.into()]
        );
    }
}
