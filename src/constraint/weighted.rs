use std::fmt;
use std::{fmt::Debug, fmt::Display};

use portgraph::{NodeIndex, SecondaryMap};

use crate::Constraint;

use super::elementary::ElementaryConstraint;
use super::{Address, ConstraintType, ConstraintVec, NodeRange, PortAddress, SpineAddress};

/// A state transition for a weighted graph trie.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
///
/// Furthermore, a constraint can include a condition on the target node weight
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeightedAdjConstraint<N> {
    /// The port is not linked.
    Dangling,
    /// The port is linked to a node satisfying the constraints.
    Link(ConstraintVec<N>),
}
impl<N: Clone + Eq> WeightedAdjConstraint<N> {
    pub(crate) fn dangling() -> Self {
        Self::Dangling
    }

    pub(crate) fn link(addr: Address, no_addr: Vec<NodeRange>, weight: N) -> Self {
        let Address {
            addr,
            label,
            // no_addr,
        } = addr;
        let constraints = no_addr
            .into_iter()
            .map(ElementaryConstraint::NoMatch)
            .chain([ElementaryConstraint::NodeWeight(weight)])
            .chain([ElementaryConstraint::PortLabel(label)])
            .chain([ElementaryConstraint::Match(addr)])
            .collect();
        Self::Link(constraints)
    }
}

impl<N: Clone + Ord + Eq + 'static> Constraint for WeightedAdjConstraint<N> {
    type Graph<'g> = (
        &'g portgraph::PortGraph,
        &'g SecondaryMap<NodeIndex, N>,
        NodeIndex,
    );

    fn is_satisfied<'g, A>(&self, ports: &A, g @ (graph, weights, root): Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>,
    {
        let ports = ports.ports(g);
        match self {
            WeightedAdjConstraint::Dangling => !ports.is_empty(),
            WeightedAdjConstraint::Link(constraints) => {
                ports.into_iter().filter_map(|p| g.0.port_link(p)).any(|p| {
                    let node = graph.port_node(p).expect("invalid port");
                    constraints.is_satisfied(p, graph, root, weights.get(node))
                })
            }
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

impl<N: Debug> Display for WeightedAdjConstraint<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dangling => write!(f, "dangling"),
            Self::Link(c) => write!(f, "{:?}", c),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum WeightedAdjConstraintType<N> {
    NodeWeight(N),
    PortLabel,
    Match,
    NoMatch(SpineAddress),
}

impl<N: Clone + Ord + 'static> ConstraintType for WeightedAdjConstraint<N> {
    type CT = WeightedAdjConstraintType<N>;

    fn constraint_type(&self) -> Self::CT {
        match self {
            Self::Dangling => WeightedAdjConstraintType::PortLabel,
            Self::Link(ConstraintVec::Vec(c)) if c.len() == 1 => match &c[0] {
                ElementaryConstraint::NodeWeight(w) => {
                    WeightedAdjConstraintType::NodeWeight(w.clone())
                }
                ElementaryConstraint::PortLabel(_) => WeightedAdjConstraintType::PortLabel,
                ElementaryConstraint::NoMatch(addr) => {
                    WeightedAdjConstraintType::NoMatch(addr.spine.clone())
                }
                ElementaryConstraint::Match(_) => WeightedAdjConstraintType::Match,
            },
            _ => panic!("Not an elementary constraint"),
        }
    }
}
