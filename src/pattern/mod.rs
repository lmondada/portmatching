//! Patterns for graph matching.
//!
//! Patterns are graphs that can be matched against other graphs.

mod unweighted;

use std::collections::BTreeMap;

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};
pub use unweighted::UnweightedPattern;

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
    fn get_boundary(&self, root: NodeIndex, graph: &PortGraph) -> PatternBoundaries {
        let mut out_edges = Vec::new();
        let mut in_edges = Vec::new();
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
                    match &port_offset {
                        PortOffset::Incoming(_) => in_edges.push(port_out),
                        PortOffset::Outgoing(_) => out_edges.push(port_out),
                    }
                }
            }
        }
        PatternBoundaries {
            _in_edges: in_edges,
            _out_edges: out_edges,
        }
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
