#![allow(dead_code)] // Temporary until refactor is complete

use std::fmt::Debug;

use derive_more::{From, Into};
use itertools::Itertools;
use petgraph::{
    algo::toposort,
    dot::Dot,
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableDiGraph,
    visit::{Bfs, EdgeRef},
    Direction,
};

use crate::{utils::online_toposort, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum TracerNodeWeight<NodeId> {
    State(NodeId),
    Terminal,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TracerEdgeWeight<EdgeId, Action> {
    ExistingEdge(EdgeId),
    TerminalEdge { action: Action },
}

/// A handle to a traced node.
///
/// A handle to such a node is obtained by tracing edge traversals from another
/// traced node (or the initial traced node).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct TracedNode(NodeIndex);

/// A handle to a traced edge.
///
/// A handle to such an edge is obtained by tracing an edge traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct TracedEdge(EdgeIndex);

/// Trace graph traversal and modifications.
///
/// Starting from an `initial_node`, keep track of edges traversed. This creates
/// traced nodes and traced paths through the graph starting at `initial_node`.
/// Graph modifications can be added on traced nodes for delayed execution, until
/// after a larger graph operation is completed.
///
/// This is useful in state automata, where graph modifications must take care not
/// to change states reachable along non-traced paths. By batching graph
/// modifications, we can reduce the number of state cloning that are required
/// and thus keep the overall state automaton smaller (see [Tracer::zip]).
///
/// There is always a graph homomorphism from the traced graph into the underlying
/// graph. The image of a traced node is the `NodeId` stored as weight. However,
/// this map is not injective: multiple traversals of the same path in the
/// underlying graph will result in multiple traced paths in the traced graph.
///
/// Actions are stored on edges that connect the action the nodes applies on and
/// a terminal node.
///
/// # Generic parameters
/// - `NodeId`: The type of node identifiers in the underlying graph.
/// - `EdgeId`: The type of edge identifiers in the underlying graph.
/// - `Action`: The type of actions that can be traced.
pub struct Tracer<NodeId, EdgeId, Action> {
    /// The edges that have been traversed and traced
    ///
    /// The only edges with a `NewEdge` weight are the ones with target at the terminal.
    trace: StableDiGraph<TracerNodeWeight<NodeId>, TracerEdgeWeight<EdgeId, Action>>,
    /// The initial node, from which all other nodes must be reachable.
    initial_node: TracedNode,
    /// The terminal node, to which all `NewEdge` connect to.
    terminal_node: TracedNode,
}

impl<NodeId, EdgeId, Action> Tracer<NodeId, EdgeId, Action>
where
    NodeId: Copy + Eq + Ord,
    EdgeId: Copy + Eq + Ord,
    Action: Eq + Clone,
{
    /// Create a new tracer for paths starting in `initial_node`.
    pub fn new(initial_node: NodeId) -> Self {
        let mut trace = StableDiGraph::new();
        let initial_node = trace.add_node(TracerNodeWeight::State(initial_node)).into();
        let terminal_node = trace.add_node(TracerNodeWeight::Terminal).into();
        Tracer {
            trace,
            initial_node,
            terminal_node,
        }
    }

    /// Handle to the initial node in the tracer.
    pub fn initial_node(&self) -> TracedNode {
        self.initial_node
    }

    /// Iterate over all nodes in the traced graph in topological order.
    pub fn toposort(&self) -> impl Iterator<Item = TracedNode> {
        toposort(&self.trace, None)
            .unwrap()
            .into_iter()
            .map(|n| n.into())
    }

    /// The node underlying a traced handle.
    pub fn node(&self, node: TracedNode) -> Option<NodeId> {
        match *self.trace.node_weight(node.into())? {
            TracerNodeWeight::State(s) => Some(s),
            TracerNodeWeight::Terminal => None,
        }
    }

    /// All traced incoming edges to a traced node.
    pub fn traced_incoming(&self, node: TracedNode) -> impl Iterator<Item = EdgeId> + '_ {
        self.trace
            .edges_directed(node.into(), Direction::Incoming)
            .map(|e| self.edge(e.id().into()).unwrap())
    }

    /// Consume the tracer and return all traced edge additions.
    pub fn into_actions(mut self) -> impl Iterator<Item = (NodeId, Action)> {
        let edge_adds = self
            .trace
            .edges_directed(self.terminal_node.into(), Direction::Incoming)
            .map(|e| e.id())
            .collect_vec();
        edge_adds.into_iter().map(move |edge_id| {
            let from_node = self.trace.edge_endpoints(edge_id).unwrap().0;
            let from = self.node(from_node.into()).unwrap();
            let TracerEdgeWeight::TerminalEdge { action } =
                self.trace.remove_edge(edge_id).unwrap()
            else {
                panic!("Non-action edge connected to terminal node");
            };
            (from, action)
        })
    }

    /// The edge underlying a traced handle.
    pub fn edge(&self, edge: TracedEdge) -> Option<EdgeId> {
        match *self.trace.edge_weight(edge.into())? {
            TracerEdgeWeight::ExistingEdge(t) => Some(t),
            TracerEdgeWeight::TerminalEdge { .. } => None,
        }
    }

    /// Trace an edge traversal and return the target as a new traced node.
    pub fn traverse_edge(
        &mut self,
        transition: EdgeId,
        from_node: TracedNode,
        to_state: NodeId,
    ) -> TracedNode {
        let to_node = self.trace.add_node(TracerNodeWeight::State(to_state));
        self.trace.add_edge(
            from_node.into(),
            to_node,
            TracerEdgeWeight::ExistingEdge(transition),
        );
        to_node.into()
    }

    /// Trace an edge addition to an untraced node.
    ///
    /// Use this to delay edge addition in the underlying graph. Recover all
    /// traced additions at the end of tracing using [`into_edge_additions`](Self::into_edge_additions).
    pub fn add_action(&mut self, action: Action, node: TracedNode) {
        let to_node = self.terminal_node;
        self.trace.add_edge(
            node.into(),
            to_node.into(),
            TracerEdgeWeight::TerminalEdge { action },
        );
    }

    /// Merge identical traced nodes and edges starting from terminal.
    ///
    /// The aim of this is to reduce the size of the traced graph, and in
    /// particular the number of edge additions. This is achieved by finding
    /// subgraphs of the trace that are identical (and hence map to the same
    /// subgraph of the underlying graph) and merging them.
    ///
    /// This also prunes any nodes that are not reachable from terminal, as they
    /// are irrelevant for edge additions.
    pub fn zip(&mut self) {
        // We will proceed from terminal to root (un-reverse at the end)
        self.trace.reverse();

        // First prune the graph of any node that is not reachable from terminal.
        prune_unreachable(self.terminal_node.into(), &mut self.trace);

        // Now starting from terminal, merge node where possible
        let mut traverser = online_toposort(self.terminal_node.into());
        let mut to_remove = HashSet::default();
        while let Some(node) = traverser.next(&self.trace) {
            if to_remove.contains(&node) {
                continue;
            }
            for merge_children in partition_children(node, &self.trace) {
                let mut merge_children = merge_children.into_iter();
                let Some(merge_into) = merge_children.next() else {
                    continue;
                };
                for merge_from in merge_children {
                    merge_nodes(merge_from, merge_into, &mut self.trace);
                    to_remove.insert(merge_from);
                }
            }
        }
        for node in to_remove {
            self.trace.remove_node(node);
        }

        self.trace.reverse();
    }
}

impl<NodeId, EdgeId, Action> Tracer<NodeId, EdgeId, Action> {
    #[allow(dead_code)]
    pub fn dot_string(&self) -> String
    where
        NodeId: Debug,
        EdgeId: Debug,
        Action: Debug,
    {
        format!("{:?}", Dot::new(&self.trace))
    }
}

impl<NodeId: Debug, EdgeId: Debug, Action: Debug> Debug for Tracer<NodeId, EdgeId, Action>
where
    NodeId: Debug,
    EdgeId: Debug,
    Action: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dot_string())
    }
}

/// Partition the children of `node` into sets of identical nodes.
///
/// Two nodes are "identical" if they have the same node weight and
/// the same set of incoming edges, each with identical edge source and edge weight.
fn partition_children<N: Eq + Ord, E: Eq>(
    node: NodeIndex,
    graph: &StableDiGraph<N, E>,
) -> Vec<Vec<NodeIndex>> {
    let key_of = |child| {
        (
            graph.node_weight(child).unwrap(),
            graph
                .edges_directed(child, Direction::Incoming)
                .map(|e| (graph.edge_weight(e.id()).unwrap(), e.source()))
                .sorted_by_key(|&(_, k)| k)
                .collect_vec(),
        )
    };
    graph
        .neighbors_directed(node, Direction::Outgoing)
        // This sorting may miss some of the grouping possibilities, but the tradeoff
        // would be to require E: Ord
        .sorted_by_key(|child| key_of(*child).0)
        .group_by(|child| key_of(*child))
        .into_iter()
        .map(|(_, indices)| indices.unique().collect_vec())
        .collect_vec()
}

/// Adds outgoing edges of `merge_from` to `merge_into`.
///
/// Does not remove any edges or vertices to not mess up the toposort traversal.
/// The caller should remove the `merge_from` node when appropriate.
fn merge_nodes<N: Eq, E: Eq + Clone>(
    merge_from: NodeIndex,
    merge_into: NodeIndex,
    graph: &mut StableDiGraph<N, E>,
) {
    let from_out = graph
        .edges_directed(merge_from, Direction::Outgoing)
        .map(|e| (e.id(), e.target()))
        .collect_vec();
    for (edge_id, edge_target) in from_out {
        let edge_weight = graph.edge_weight(edge_id).unwrap().clone();
        graph.add_edge(merge_into, edge_target, edge_weight);
    }
}

fn prune_unreachable<N, E>(node: NodeIndex, graph: &mut StableDiGraph<N, E>) {
    let mut visited = HashSet::default();
    let mut bfs = Bfs::new(&*graph, node);
    while let Some(node) = bfs.next(&*graph) {
        visited.insert(node);
    }
    let mut to_remove = graph.node_indices().collect_vec();
    to_remove.retain(|n| !visited.contains(n));

    for node in to_remove {
        graph.remove_node(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::{fixture, rstest};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestAddEdge(usize);

    #[fixture]
    fn tracer() -> Tracer<usize, (), TestAddEdge> {
        // Trace paths starting at 0 (of some graph we don't build)
        let mut tracer = Tracer::new(0);
        // Traverse 0 -> 2 -> 3, then add 3 -> 6
        let traced_2 = tracer.traverse_edge((), tracer.initial_node(), 2);
        let traced_3 = tracer.traverse_edge((), traced_2, 3);
        tracer.add_action(TestAddEdge(6), traced_3);
        // Using the same 0 -> 2, then traverse 2 -> 5, then add 5 -> 66
        let traced_5 = tracer.traverse_edge((), traced_2, 5);
        tracer.add_action(TestAddEdge(66), traced_5);
        // New traversal 0 -> 1 -> 2 -> 3, then add 3 -> 6
        let traced_1 = tracer.traverse_edge((), tracer.initial_node(), 1);
        let traced_2 = tracer.traverse_edge((), traced_1, 2);
        let traced_3 = tracer.traverse_edge((), traced_2, 3);
        tracer.add_action(TestAddEdge(6), traced_3);
        // New traversal 0 -> 7, no edge addition
        tracer.traverse_edge((), tracer.initial_node(), 7);
        tracer
    }

    #[fixture]
    fn tracer_parallel() -> Tracer<usize, (), TestAddEdge> {
        // Trace paths starting at 0 (of some graph we don't build)
        let mut tracer = Tracer::new(0);
        // Traverse 0 -> 1, then add 1 -> 2
        let traced_1 = tracer.traverse_edge((), tracer.initial_node(), 1);
        tracer.add_action(TestAddEdge(2), traced_1);
        // Traverse 0 -> 1, then add 1 -> 2 (again)
        let traced_1 = tracer.traverse_edge((), tracer.initial_node(), 1);
        tracer.add_action(TestAddEdge(2), traced_1);

        tracer
    }

    #[rstest]
    fn test_tracer(tracer: Tracer<usize, (), TestAddEdge>) {
        insta::assert_snapshot!(tracer.dot_string());
    }

    #[rstest]
    fn test_tracer_zip(mut tracer: Tracer<usize, (), TestAddEdge>) {
        assert_eq!(tracer.trace.node_count(), 9);
        tracer.zip();
        assert_eq!(tracer.trace.node_count(), 7);
        insta::assert_snapshot!(tracer.dot_string());
    }

    #[rstest]
    fn test_zip_parallel(mut tracer_parallel: Tracer<usize, (), TestAddEdge>) {
        assert_eq!(tracer_parallel.trace.node_count(), 4);
        tracer_parallel.zip();
        assert_eq!(tracer_parallel.trace.node_count(), 3);
        insta::assert_snapshot!(tracer_parallel.dot_string());
    }
}
