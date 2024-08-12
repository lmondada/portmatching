use petgraph::{
    algo::{toposort, Cycle},
    visit::{
        depth_first_search, Control, DfsEvent, GraphBase, IntoNeighborsDirected,
        IntoNodeIdentifiers, Reversed, Visitable,
    },
    Direction,
};
use std::hash::Hash;

use crate::{HashMap, HashSet};

/// A map from nodes to their depth in a graph.
///
/// Used to speed up checking whether adding an edge would create a cycle.
pub(super) struct NodeDepthCache<N> {
    /// Store node depths
    ///
    /// If u -> v is an edge then the depth[u] is strictly smaller than depth[v].
    depths: HashMap<N, usize>,
}

impl<N: Eq + Hash + Copy> NodeDepthCache<N> {
    pub fn with_graph<G>(graph: G) -> Result<Self, Cycle<G::NodeId>>
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        let mut depths = HashMap::default();
        for node in toposort(graph, None)? {
            let depth = graph
                .neighbors_directed(node, Direction::Incoming)
                .map(|n| depths[&n])
                .max()
                .map_or(0, |d| d + 1);
            depths.insert(node, depth);
        }
        Ok(Self { depths })
    }

    /// Whether `node` is an ancestor of any node in `preds` in the graph.
    ///
    /// If `reverse` is true, check whether `node` is a descendant instead.
    pub fn is_ancestor<G>(&self, node: N, preds: &HashSet<N>, graph: G, reverse: bool) -> bool
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        let depth_threshold = if reverse {
            preds.iter().map(|p| self.depths[p]).min()
        } else {
            preds.iter().map(|p| self.depths[p]).max()
        };
        let Some(depth_threshold) = depth_threshold else {
            return false;
        };
        let past_threshold = |d| {
            if reverse {
                d <= depth_threshold
            } else {
                d >= depth_threshold
            }
        };
        let event_handler = |event| {
            if let DfsEvent::Discover(n, _) = event {
                if preds.contains(&n) {
                    Control::Break(true)
                } else if past_threshold(self.depths[&n]) {
                    Control::Prune
                } else {
                    Control::Continue
                }
            } else {
                Control::Continue
            }
        };
        let dfs_res = if reverse {
            depth_first_search(Reversed(graph), Some(node), event_handler)
        } else {
            depth_first_search(graph, Some(node), event_handler)
        };
        matches!(dfs_res, Control::Break(true))
    }

    /// Whether there is a path between `node1` and `node2` in the graph.
    pub fn path_exists<G>(&self, node: N, others: &HashSet<N>, graph: G) -> bool
    where
        G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + GraphBase<NodeId = N>,
    {
        self.is_ancestor(node, others, graph, false) || self.is_ancestor(node, others, graph, true)
    }

    /// Update node depths incurred by merging `nodes`.
    ///
    /// This will update `node.first()` to the maximum of all node depths in
    /// `nodes`. Other nodes in `nodes` are removed.
    ///
    /// Just like merging, this assumes that all nodes are space-like separated
    /// (i.e. no path from one to another).
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
        let checker = NodeDepthCache::with_graph(&graph).unwrap();

        // Test is_descendant
        assert!(checker.is_ancestor(n3, &HashSet::from_iter([n1]), &graph, true));
        assert!(checker.is_ancestor(n3, &HashSet::from_iter([n2]), &graph, true));
        assert!(checker.is_ancestor(n3, &HashSet::from_iter([n4]), &graph, true));
        assert!(checker.is_ancestor(n2, &HashSet::from_iter([n1]), &graph, true));
        assert!(checker.is_ancestor(n4, &HashSet::from_iter([n1]), &graph, true));

        assert!(checker.is_ancestor(n1, &HashSet::from_iter([n2]), &graph, false));
        assert!(checker.is_ancestor(n1, &HashSet::from_iter([n3]), &graph, false));
        assert!(checker.is_ancestor(n1, &HashSet::from_iter([n4]), &graph, false));
        assert!(checker.is_ancestor(n2, &HashSet::from_iter([n3]), &graph, false));
        assert!(checker.is_ancestor(n4, &HashSet::from_iter([n3]), &graph, false));

        // Test with cyclic graph
        graph.add_edge(n3, n1, ());
        assert!(NodeDepthCache::with_graph(&graph).is_err());
    }
}
