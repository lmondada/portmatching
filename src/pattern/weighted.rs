use portgraph::{NodeIndex, PortGraph, SecondaryMap};

use crate::{constraint::WeightedConstraint, Pattern, UnweightedPattern};

use super::{Edge, InvalidPattern};

/// A pattern graph along with node weights.
///
/// Patterns must be connected and have a fixed `root` node,
/// which by default is chosen to be the centre of the graph, for fast
/// matching and short relative paths to the root.
pub struct WeightedPattern<N> {
    /// The (unweighted) pattern
    pattern: UnweightedPattern,
    /// Node weights
    pub(crate) weights: SecondaryMap<NodeIndex, N>,
}

impl<N: Clone> Pattern for WeightedPattern<N> {
    type Constraint = WeightedConstraint<N>;

    fn graph(&self) -> &PortGraph {
        self.pattern.graph()
    }

    fn root(&self) -> NodeIndex {
        self.pattern.root()
    }

    fn to_constraint(&self, e @ Edge(_, in_port): &Edge) -> Self::Constraint {
        self.pattern
            .to_constraint(e)
            .to_weighted(in_port.map(|in_port| {
                let node = self.graph().port_node(in_port).expect("invalid port");
                self.weights[node].clone()
            }))
    }

    fn all_lines(&self) -> Vec<Vec<Edge>> {
        self.pattern.all_lines()
    }
}

impl<N> WeightedPattern<N> {
    /// Create a new pattern from a graph.
    pub fn from_weighted_graph(
        graph: PortGraph,
        weights: SecondaryMap<NodeIndex, N>,
    ) -> Result<Self, InvalidPattern> {
        Ok(WeightedPattern {
            pattern: UnweightedPattern::from_graph(graph)?,
            weights,
        })
    }
}
