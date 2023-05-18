use std::fmt::{self, Display};

use portgraph::{NodeIndex, PortGraph};

use super::{Address, Constraint, ConstraintVec, ElementaryConstraint, PortAddress};

/// Adjacency constraint for unweighted graphs.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnweightedAdjConstraint {
    Dangling,
    Link(ConstraintVec<()>),
}

impl UnweightedAdjConstraint {
    pub(crate) fn dangling() -> Self {
        Self::Dangling
    }

    pub(crate) fn link(port_addr: Address) -> Self {
        let Address {
            addr,
            label,
            no_addr,
        } = port_addr;
        let constraints = no_addr
            .into_iter()
            .map(ElementaryConstraint::NoMatch)
            .chain([ElementaryConstraint::PortLabel(label)])
            .chain([ElementaryConstraint::Match(addr)])
            .collect();
        Self::Link(constraints)
    }
}

impl Constraint for UnweightedAdjConstraint {
    type Graph<'g> = (&'g PortGraph, NodeIndex);

    fn is_satisfied<'g, A>(&self, ports: &A, g: Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>,
    {
        let ports = ports.ports(g);
        match self {
            UnweightedAdjConstraint::Dangling => !ports.is_empty(),
            UnweightedAdjConstraint::Link(constraints) => ports
                .into_iter()
                .filter_map(|p| g.0.port_link(p))
                .any(|p| constraints.is_satisfied(p, g.0, g.1, &())),
        }
    }

    fn and(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Dangling, Self::Dangling) => Some(Self::Dangling),
            (Self::Dangling, Self::Link(_)) => Some(other.clone()),
            (Self::Link(_), Self::Dangling) => Some(self.clone()),
            (Self::Link(c1), Self::Link(c2)) => c1.and(c2).map(Self::Link),
        }
    }

    fn to_elementary(&self) -> Vec<Self>
    where
        Self: Clone,
    {
        let Self::Link(c) = self else {
            return vec![self.clone()];
        };
        let ConstraintVec::Vec(c) = c else {
            return vec![]
        };
        c.iter()
            .cloned()
            .map(|c| Self::Link(ConstraintVec::new(vec![c])))
            .collect()
    }
}

impl Display for UnweightedAdjConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dangling => write!(f, "dangling"),
            Self::Link(c) => write!(f, "{:?}", c),
        }
    }
}
