use std::{fmt::Debug, fmt::Display, mem};

use portgraph::{NodeIndex, PortOffset, SecondaryMap};

use crate::Constraint;

use super::{
    unweighted::{adjacency_constraint, simplify_constraints, Address, NodeAddress, PortLabel},
    PortAddress, UnweightedConstraint,
};

/// A state transition for a weighted graph trie.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
///
/// Furthermore, a constraint can include a condition on the target node weight
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeightedConstraint<N> {
    // All constraints must be satisfied
    AllAdjacencies {
        label: PortLabel,
        target_weight: N,
        no_match: Vec<(Vec<PortOffset>, usize, [isize; 2])>,
        the_matches: Vec<(Vec<PortOffset>, usize, isize)>,
    },
    // Port must be linked to one of `other_ports`
    Adjacency {
        target_weight: N,
        other_ports: Address,
    },
    // Port must be dangling (at least existing)
    Dangling,
}

impl<N: Clone> WeightedConstraint<N> {
    fn to_vec(&self) -> Vec<Self> {
        let WeightedConstraint::AllAdjacencies {
            label,
            no_match,
            the_matches,
            target_weight,
        } = self else { return vec![self.clone()] };
        let mut no_match = no_match.clone();
        the_matches
            .iter()
            .map(|the_match| WeightedConstraint::Adjacency {
                other_ports: Address {
                    addr: NodeAddress {
                        no_match: mem::take(&mut no_match),
                        the_match: the_match.clone(),
                    },
                    label: *label,
                },
                target_weight: target_weight.clone(),
            })
            .collect()
    }

    fn to_unweighted(&self) -> UnweightedConstraint {
        match self.clone() {
            WeightedConstraint::AllAdjacencies {
                label,
                no_match,
                the_matches,
                ..
            } => UnweightedConstraint::AllAdjacencies {
                label,
                no_match,
                the_matches,
            },
            WeightedConstraint::Adjacency { other_ports, .. } => {
                UnweightedConstraint::Adjacency { other_ports }
            }
            WeightedConstraint::Dangling => UnweightedConstraint::Dangling,
        }
    }

    fn target_weight(&self) -> Option<&N> {
        match self {
            WeightedConstraint::Adjacency { target_weight, .. } => Some(target_weight),
            WeightedConstraint::Dangling => None,
            WeightedConstraint::AllAdjacencies { target_weight, .. } => Some(target_weight),
        }
    }
}

impl<N: Clone + Ord + Eq + 'static> Constraint for WeightedConstraint<N> {
    type Graph<'g> = (
        &'g portgraph::PortGraph,
        &'g SecondaryMap<NodeIndex, N>,
        NodeIndex,
    );

    fn is_satisfied<'g, A>(&self, this_ports: &A, g @ (graph, weights, _): Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>,
    {
        match self {
            WeightedConstraint::Adjacency {
                target_weight,
                other_ports,
            } => {
                let other_ports = other_ports.ports(g);
                let this_ports = this_ports.ports(g);
                let mut ports = adjacency_constraint(graph, this_ports, other_ports);
                ports.any(|p| {
                    let node = graph.port_node(p).expect("invalid port");
                    weights[node] == *target_weight
                })
            }
            WeightedConstraint::Dangling => !this_ports.ports(g).is_empty(),
            WeightedConstraint::AllAdjacencies { .. } => {
                self.to_vec().iter().all(|c| c.is_satisfied(this_ports, g))
            }
        }
    }

    fn and(&self, other: &Self) -> Option<Self> {
        let t1 = self.target_weight();
        let t2 = other.target_weight();
        if let (Some(t1), Some(t2)) = (t1, t2) {
            if t1 != t2 {
                return None;
            }
        }
        let target_weight = t1.or(t2).cloned();

        simplify_constraints(vec![self.to_unweighted(), other.to_unweighted()])
            .map(|c| c.to_weighted(target_weight))
    }
}

impl<N: Debug> Display for WeightedConstraint<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightedConstraint::AllAdjacencies {
                the_matches,
                target_weight,
                ..
            } => {
                write!(
                    f,
                    "{:?}, All({})",
                    target_weight,
                    the_matches
                        .iter()
                        .map(|(_, i, j)| format!("({i}, {j}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            WeightedConstraint::Adjacency {
                other_ports,
                target_weight,
            } => {
                write!(
                    f,
                    "{:?} Adjacency({:?}, {:?})",
                    target_weight, other_ports.addr.the_match, other_ports.label
                )
            }
            WeightedConstraint::Dangling => write!(f, "Dangling"),
        }
    }
}
