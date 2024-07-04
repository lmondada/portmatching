use petgraph::{
    algo::{toposort, Cycle},
    visit::{
        depth_first_search, Control, DfsEvent, GraphBase, IntoNeighborsDirected,
        IntoNodeIdentifiers, Visitable,
    },
    Direction,
};
use std::hash::Hash;

use crate::HashMap;

/// Speed up checking whether adding an edge would create a cycle
pub(super) struct AcyclicChecker<N> {
    /// Store node depths
    ///
    /// If u -> v is an edge then the depth[u] is strictly smaller than depth[v].
    depths: HashMap<N, usize>,
}

impl<N: Eq + Hash + Copy> AcyclicChecker<N> {
    pub fn with_graph<G>(graph: G) -> Result<Self, Cycle<G::NodeId>>
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        let mut depths = HashMap::default();
        for node in toposort(graph, None)? {
            let max_prev_depth = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|n| depths[&n])
                .max();
            let depth = if let Some(max_prev_depth) = max_prev_depth {
                max_prev_depth + 1
            } else {
                0
            };
            depths.insert(node, depth);
        }
        Ok(Self { depths })
    }

    /// Whether `node` is a descendant of `pred` in the graph.
    pub fn is_descendant<G>(&self, node: N, pred: N, graph: G) -> bool
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        if self.depths[&node] <= self.depths[&pred] {
            return false;
        }
        let dfs_res = depth_first_search(graph, Some(pred), |event| {
            if let DfsEvent::Discover(n, _) = event {
                if n == node {
                    Control::Break(true)
                } else if self.depths[&n] >= self.depths[&node] {
                    Control::Prune
                } else {
                    Control::Continue
                }
            } else {
                Control::Continue
            }
        });
        matches!(dfs_res, Control::Break(true))
    }

    /// Whether there is a path between `node1` and `node2` in the graph.
    pub fn path_exists<G>(&self, node1: N, node2: N, graph: G) -> bool
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        self.is_descendant(node1, node2, graph) || self.is_descendant(node2, node1, graph)
    }

    /// Merge all `nodes` into the first node in the slice.
    pub fn merge_nodes(&mut self, nodes: impl Iterator<Item = N>) {
        let mut nodes = nodes.peekable();
        let Some(&first_node) = nodes.peek() else {
            return;
        };
        let max_depth = nodes.flat_map(|n| self.depths.remove(&n)).max().unwrap();
        self.depths.insert(first_node, max_depth);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use petgraph::graph::Graph;

    #[test]
    fn test_acyclic_checker() {
        // Create a simple acyclic graph
        let mut graph = Graph::<(), ()>::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n1, n4, ());
        graph.add_edge(n4, n3, ());

        // Create AcyclicChecker
        let checker = AcyclicChecker::with_graph(&graph).unwrap();

        // Test is_descendant
        assert!(checker.is_descendant(n3, n1, &graph));
        assert!(checker.is_descendant(n3, n2, &graph));
        assert!(checker.is_descendant(n3, n4, &graph));
        assert!(checker.is_descendant(n2, n1, &graph));
        assert!(checker.is_descendant(n4, n1, &graph));

        assert!(!checker.is_descendant(n1, n2, &graph));
        assert!(!checker.is_descendant(n1, n3, &graph));
        assert!(!checker.is_descendant(n1, n4, &graph));
        assert!(!checker.is_descendant(n2, n3, &graph));
        assert!(!checker.is_descendant(n4, n3, &graph));

        // Test with cyclic graph
        graph.add_edge(n3, n1, ());
        assert!(AcyclicChecker::with_graph(&graph).is_err());
    }
}
