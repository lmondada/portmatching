use portgraph::PortIndex;

mod skeleton;
pub(crate) mod unweighted;

pub use skeleton::Skeleton;
pub use unweighted::UnweightedConstraint;

/// Constraints for graph trie transitions.
///
/// To each transition in a graph trie corresponds a constraint. If a constraint
/// is satisfied, then the transition is valid.
///
/// The simplest constraint is the [`UnweightedConstraint`], which only checks
/// for adjacency. More complex constraints can be defined by implementing this
/// trait, for example for weighted or hierarchical graphs.
pub trait Constraint//: Clone + PartialEq + Eq + PartialOrd + Ord
where
    Self: Sized,
{
    /// The type of the graph the constraint is applied to.
    type Graph<'g>;

    /// Check if the constraint is satisfied.
    fn is_satisfied<'g, A>(&self, ports: &A, g: Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>;

    /// Merge two constraints.
    fn and(&self, other: &Self) -> Option<Self>;
}

pub trait PortAddress<Graph>: Clone + PartialEq + Eq + PartialOrd + Ord {
    fn ports(&self, g: Graph) -> Vec<PortIndex>;

    fn port(&self, g: Graph) -> Option<PortIndex> {
        let ports = self.ports(g);
        (ports.len() == 1).then_some(ports[0])
    }
}
