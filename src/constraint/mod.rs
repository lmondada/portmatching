use portgraph::{NodeIndex, PortGraph, PortIndex};

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
pub trait Constraint: Clone + PartialEq + Eq + PartialOrd + Ord
where
    Self: Sized,
{
    /// The addressing scheme used to identify ports.
    ///
    /// Port addresses do not need to identify ports uniquely, but they must
    /// be unique when used as keys in graph trie states.
    type Address: PortAddress;

    /// Check if the constraint is satisfied.
    fn is_satisfied(&self, ports: &Self::Address, g: &PortGraph, root: NodeIndex) -> bool;

    /// Merge two constraints.
    fn and(&self, other: &Self) -> Option<Self>;
}

pub trait PortAddress: Clone + PartialEq + Eq + PartialOrd + Ord {
    fn ports(&self, g: &PortGraph, root: NodeIndex) -> Vec<PortIndex>;

    fn port(&self, g: &PortGraph, root: NodeIndex) -> Option<PortIndex> {
        let ports = self.ports(g, root);
        (ports.len() == 1).then_some(ports[0])
    }
}
