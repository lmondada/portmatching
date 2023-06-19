//! Patterns for graph matching.
//!
//! Patterns are graphs that can be matched against other graphs.

mod unweighted;
mod weighted;

use std::collections::BTreeMap;

use portgraph::{
    substitute::{BoundedSubgraph, SubgraphRef},
    NodeIndex, PortGraph, PortIndex, PortOffset,
};

#[doc(inline)]
pub use unweighted::UnweightedPattern;
#[doc(inline)]
pub use weighted::WeightedPattern;

/// An edge in a graph.
///
/// The edge might be dangling, in which case the second port index is `None`.
#[derive(Debug, PartialEq, Eq)]
pub struct Edge(pub(crate) PortIndex, pub(crate) Option<PortIndex>);

/// A pattern for graph matching.
///
/// Patterns are graphs that can be matched against other graphs. Different
/// implementations might support e.g. weighted graphs.
pub trait Pattern {
    /// The type of constraints used by this pattern.
    type Constraint;

    /// The pattern graph as a [`PortGraph`].
    fn graph(&self) -> &PortGraph;
    /// The root of the pattern in the port graph.
    fn root(&self) -> NodeIndex;

    /// Convert an edge to a constraint.
    fn to_constraint(&self, e: &Edge) -> Self::Constraint;

    /// Get a partition of the edges of the pattern.
    fn all_lines(&self) -> Vec<Vec<Edge>>;

    /// Get the boundary of this pattern in a graph.
    ///
    /// This is useful to get the location of a pattern in a graph
    /// given a mapping of its root.
    fn get_boundary(&self, root: NodeIndex, graph: &PortGraph) -> BoundedSubgraph {
        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        let mut pattern_to_graph = BTreeMap::from([(self.root(), root)]);
        for line in self.all_lines() {
            for edge in line {
                let curr_pattern = self.graph().port_node(edge.0).unwrap();
                let curr_graph = pattern_to_graph[&curr_pattern];
                let port_offset = self.graph().port_offset(edge.0).unwrap();
                let port_out = graph.port_index(curr_graph, port_offset).unwrap();
                if let Some(next_port) = edge.1 {
                    let next_pattern = self.graph().port_node(next_port).unwrap();
                    let port_in = graph.port_link(port_out).unwrap();
                    let next_graph = graph.port_node(port_in).unwrap();
                    pattern_to_graph.insert(next_pattern, next_graph);
                } else {
                    let link = graph.port_link(port_out).expect("Invalid pattern in graph");
                    match &port_offset {
                        PortOffset::Incoming(_) => inputs.push(link),
                        PortOffset::Outgoing(_) => outputs.push(link),
                    }
                }
            }
        }
        let subgraph =
            SubgraphRef::new_from_indices(pattern_to_graph.into_values(), Some(graph.node_count()));
        BoundedSubgraph::new(subgraph, inputs, outputs)
    }

    /// Every pattern has a unique canonical ordering of its edges.
    ///
    /// Pattern matching can be done by matching these edges one-by-one
    fn canonical_edge_ordering(&self) -> Vec<Edge> {
        self.all_lines().into_iter().flatten().collect()
    }
}

impl<P: Pattern + ?Sized> Pattern for Box<P> {
    type Constraint = P::Constraint;

    fn graph(&self) -> &PortGraph {
        self.as_ref().graph()
    }

    fn root(&self) -> NodeIndex {
        self.as_ref().root()
    }

    fn to_constraint(&self, e: &Edge) -> Self::Constraint {
        self.as_ref().to_constraint(e)
    }

    fn all_lines(&self) -> Vec<Vec<Edge>> {
        self.as_ref().all_lines()
    }
}

/// The boundary of a pattern in a graph.
///
/// Given as a list of in- and out-edges.
#[derive(Debug)]
pub struct PatternBoundaries {
    _in_edges: Vec<PortIndex>,
    _out_edges: Vec<PortIndex>,
}

/// Error that can occur when creating a pattern from a graph.
#[derive(Debug, PartialEq, Eq)]
pub enum InvalidPattern {
    /// A pattern must always be connected.
    DisconnectedPattern,
    /// Empty patterns are not allowed.
    EmptyPattern,
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::{
        substitute::{BoundedSubgraph, SubgraphRef},
        PortGraph,
    };

    use crate::{Pattern, UnweightedPattern};

    #[test]
    fn test_get_boundary() {
        let mut g = PortGraph::new();
        let mut p = PortGraph::new();
        //      1 -- 3
        // 0 <          > 4 -- 5
        //      2 ----
        let n0 = g.add_node(0, 2);
        let n1 = g.add_node(1, 1);
        let p1 = p.add_node(1, 1);
        g.link_nodes(n0, 0, n1, 0).unwrap();
        let n2 = g.add_node(1, 1);
        let p2 = p.add_node(1, 1);
        g.link_nodes(n0, 1, n2, 0).unwrap();
        let n3 = g.add_node(1, 1);
        let p3 = p.add_node(1, 1);
        g.link_nodes(n1, 0, n3, 0).unwrap();
        p.link_nodes(p1, 0, p3, 0).unwrap();
        let n4 = g.add_node(2, 1);
        let p4 = p.add_node(2, 1);
        g.link_nodes(n3, 0, n4, 0).unwrap();
        p.link_nodes(p3, 0, p4, 0).unwrap();
        g.link_nodes(n2, 0, n4, 1).unwrap();
        p.link_nodes(p2, 0, p4, 1).unwrap();
        let n5 = g.add_node(1, 0);
        g.link_nodes(n4, 0, n5, 0).unwrap();

        let left = UnweightedPattern::from_graph(p)
            .unwrap()
            .get_boundary(n3, &g);
        let right = BoundedSubgraph::new(
            SubgraphRef::new_from_indices([n1, n2, n3, n4], Some(6)),
            g.outputs(n0).collect(),
            g.inputs(n5).collect(),
        );
        // Ideally: assert_eq!(left, right);
        assert_eq!(
            left.subgraph.nodes().collect_vec(),
            right.subgraph.nodes().collect_vec()
        );
        assert_eq!(left.inputs, right.inputs);
        assert_eq!(left.outputs, right.outputs);
    }
}
