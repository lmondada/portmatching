use std::hash::Hash;

use itertools::Itertools;
use petgraph::{visit::IntoNeighborsDirected, Direction};

use crate::HashSet;

/// Perform a topological sort of a graph, while allowing graph mutations.
///
/// The graph is expected to be acyclic and all nodes must be reachable from the
/// root (i.e. the root is the source of the graph).
///
/// The particularity of this implemention is that graph mutations during
/// traversal are supported, as long as node indices are stable. As a tradeoff,
/// this traversal is not particularly efficient: each call to `next` may take
/// up to O(V + E) time.
///
/// ## Supported graph modifications
///
/// Edge and vertex additions are always supported, and as long as additional
/// edges do not invalidate the topological ordering of the nodes already
/// emitted, the result will be a valid topological order of the final graph.
///
/// Similarly, node and edges deletions are supported and the resulting topological
/// ordering will be a valid order of the initial graph (and thus of the final
/// graph too), as long as the edges do not disconnect the graph (more precisely
/// the connected component that the `root` is in).
pub fn online_toposort<N: Eq + Hash + Copy>(root: N) -> OnlineToposort<N> {
    OnlineToposort::new(root)
}

pub struct OnlineToposort<N> {
    /// Nodes that have been visited
    visited: HashSet<N>,
    /// Stack of nodes that were ready to be visited when last seen.
    ///
    /// Before outputting these nodes, it should be checked again that
    /// they are ready, in case the graph changed.
    ///
    /// In theory we could make do without this and recompute our options at
    /// every `next` call, but we choose to keep a short stack of valid choices
    /// to speed up successive calls.
    to_visit_next: Vec<N>,
}

impl<N: Eq + Hash + Copy> OnlineToposort<N> {
    fn new(root: N) -> Self {
        Self {
            visited: HashSet::default(),
            to_visit_next: vec![root],
        }
    }

    /// A node whose predecessors have all been visited
    fn is_ready<G: IntoNeighborsDirected<NodeId = N>>(&self, node: N, graph: &G) -> bool {
        node_is_ready(node, graph, &self.visited)
    }

    /// Remove previously ready nodes that are no longer ready (due to changes to the graph)
    fn prune_unready<G: IntoNeighborsDirected<NodeId = N>>(&mut self, graph: &G) {
        let Self {
            visited,
            to_visit_next,
        } = self;
        to_visit_next.retain(|&n| node_is_ready(n, graph, visited));
    }

    /// A node that is either queued to be visited, or has been visited
    fn is_queued_or_visited(&self, node: N) -> bool {
        self.to_visit_next.contains(&node) || self.visited.contains(&node)
    }

    /// Get the next node to visit, or None if no unvisited node can be reached.
    pub fn next<G: IntoNeighborsDirected<NodeId = N>>(&mut self, graph: G) -> Option<N> {
        // Make sure our ready nodes are still currently ready
        self.prune_unready(&graph);

        // Refill supply of ready nodes if empty
        let mut all_visited = self.visited.iter().copied();
        while self.to_visit_next.is_empty() {
            let ready_nodes = graph
                .neighbors_directed(all_visited.next()?, Direction::Outgoing)
                .filter(|&n| self.is_ready(n, &graph))
                .filter(|&n| !self.is_queued_or_visited(n))
                .unique()
                .collect_vec();
            self.to_visit_next.extend(ready_nodes);
        }

        let next_state = self.to_visit_next.pop()?;
        self.visited.insert(next_state);
        Some(next_state)
    }
}

fn node_is_ready<N, G>(node: N, graph: G, visited: &HashSet<N>) -> bool
where
    N: Eq + Hash + Copy,
    G: IntoNeighborsDirected<NodeId = N>,
{
    graph
        .neighbors_directed(node, Direction::Incoming)
        .all(|p| visited.contains(&p))
}

#[cfg(test)]
mod tests {
    use std::iter::from_fn;

    use itertools::Itertools;
    use petgraph::{stable_graph::StableDiGraph, visit::EdgeRef};
    use rstest::{fixture, rstest};

    use super::*;

    /// 0 -> 2 -> 3 -> 1
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

    #[rstest]
    fn test_online_toposort_with_live_changes(mut graph: StableDiGraph<usize, ()>) {
        let mut toposort = online_toposort(0.into());
        assert_eq!(
            from_fn(|| toposort.next(&graph)).take(3).collect_vec(),
            vec![0.into(), 2.into(), 3.into()]
        );

        // Now delete edge 3 -> 1 and instead add edge 2 -> 1
        let edge_3_1 = graph
            .edges_connecting(3.into(), 1.into())
            .exactly_one()
            .unwrap();
        graph.remove_edge(edge_3_1.id());
        graph.add_edge(2.into(), 1.into(), ());
        assert_eq!(toposort.next(&graph), Some(1.into()));

        // And finally, create a new node and link it to 0
        let node_4 = graph.add_node(4);
        graph.add_edge(0.into(), node_4, ());
        assert_eq!(toposort.next(&graph), Some(4.into()));
        assert_eq!(toposort.next(&graph), None);
    }

    #[test]
    fn test_disconnected() {
        let mut graph = StableDiGraph::<usize, ()>::new();
        (0..3).for_each(|i| {
            graph.add_node(i);
        });
        graph.add_edge(0.into(), 1.into(), ());
        let mut toposort = online_toposort(2.into());
        assert_eq!(toposort.next(&graph), Some(2.into()));
        assert_eq!(toposort.next(&graph), None);
    }

    #[test]
    fn test_parallel() {
        let mut graph = StableDiGraph::<usize, ()>::new();
        (0..2).for_each(|i| {
            graph.add_node(i);
        });
        graph.add_edge(0.into(), 1.into(), ());
        graph.add_edge(0.into(), 1.into(), ());
        let mut toposort = online_toposort(0.into());
        assert_eq!(
            from_fn(|| toposort.next(&graph)).collect_vec(),
            vec![0.into(), 1.into()]
        );
    }
}
