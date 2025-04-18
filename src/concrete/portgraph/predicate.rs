//! Predicates (the core of constraints), for port graph matching.

use std::{borrow::Borrow, collections::BTreeSet, fmt::Debug};

use itertools::Itertools;
use portgraph::{LinkView, NodeIndex, PortGraph, PortOffset, SecondaryMap, UnmanagedDenseMap};

use crate::{
    constraint::{
        tag::{ConstraintTag, Tag},
        ArityPredicate, DeterministicConstraintEvaluator,
    },
    pattern::Satisfiable,
    ConditionalPredicate, Constraint, EvaluatePredicate,
};

use super::{indexing::PGIndexKey, PGConstraint};

/// A predicate for constraints on (weighted) port graph
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PGPredicate<NodeWeight = ()> {
    /// The node has a specific weight
    ///
    /// Predicates applying to the same node are mutually exclusive.
    HasNodeWeight(NodeWeight),
    /// The two nodes are connected by an edge from `left_port` to `right_port`
    ///
    /// Predicates applying to the same node and port are mutually exclusive.
    IsConnected {
        /// The left port of the edge
        left_port: PortOffset,
        /// The right port of the edge
        right_port: PortOffset,
    },
    /// The node is not equal to any of the other `n_other` nodes
    ///
    /// For simplicity, never mutually exclusive
    IsNotEqual {
        /// The number of other nodes, determining the predicate arity.
        n_other: usize,
    },
}

/// Label constraints with tags for efficient constraint evaluation
/// and pattern matcher construction.
///
/// Constraints labelled with the same [`PGTag::HasNodeWeight`] or
/// [`PGTag::IsConnected`] tag are always mutually exclusive.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PGTag {
    /// Constraint tag for [`PGPredicate::HasNodeWeight`] predicates.
    ///
    /// Any two [`PGPredicate::HasNodeWeight`] predicates on the same node
    /// (i.e. with the same tag) are mutually exclusive.
    HasNodeWeight(PGIndexKey),
    /// Constraint tag for [`PGPredicate::IsConnected`] predicates.
    ///
    /// Any two [`PGPredicate::IsConnected`] predicates on the same port & node
    /// (i.e. with the same tag) are mutually exclusive.
    IsConnected(PGIndexKey, PortOffset),
    /// Constraint tag for [`PGPredicate::IsNotEqual`] predicates.
    ///
    /// This is non-deterministic, i.e. no mutual exclusivity between constraints
    /// with the same tag.
    IsNotEqual(PGIndexKey),
}

impl<W> PGPredicate<W> {
    /// Whether the predicate is satisfied, given graph, node weights and arguments
    fn is_satisfied(
        &self,
        graph: &PortGraph,
        node_weights: &impl SecondaryMap<NodeIndex, W>,
        args: &[impl Borrow<NodeIndex>],
    ) -> Result<bool, String>
    where
        W: Eq,
    {
        match self {
            PGPredicate::HasNodeWeight(weight) => {
                if let [node] = args {
                    Ok(node_weights.get(*node.borrow()) == weight)
                } else {
                    Err("expected args of length 1 (arity 1)".to_string())
                }
            }
            &PGPredicate::IsConnected {
                left_port,
                right_port,
            } => {
                if let [left, right] = args {
                    Ok(has_edge(
                        graph,
                        *left.borrow(),
                        left_port,
                        *right.borrow(),
                        right_port,
                    ))
                } else {
                    Err("expected args of length 2 (arity 2)".to_string())
                }
            }
            &PGPredicate::IsNotEqual { n_other } => {
                if let [node, other @ ..] = args {
                    let other = other.iter().map(|n| *n.borrow()).collect_vec();
                    Ok(!other.contains(node.borrow()))
                } else {
                    Err(format!("expected args of length {}", n_other + 1))
                }
            }
        }
    }

    fn get_tags(&self, keys: &[PGIndexKey]) -> Vec<PGTag> {
        use PGPredicate::*;
        match self {
            HasNodeWeight(_) => {
                vec![PGTag::HasNodeWeight(keys[0])]
            }
            IsConnected {
                left_port,
                right_port,
            } => vec![
                PGTag::IsConnected(keys[0], *left_port),
                PGTag::IsConnected(keys[1], *right_port),
            ],
            IsNotEqual { .. } => {
                vec![PGTag::IsNotEqual(keys[0])]
            }
        }
    }
}

impl<W: Clone + Ord> ArityPredicate for PGPredicate<W> {
    fn arity(&self) -> usize {
        match self {
            PGPredicate::HasNodeWeight(..) => 1,
            PGPredicate::IsConnected { .. } => 2,
            PGPredicate::IsNotEqual { n_other } => n_other + 1,
        }
    }
}

impl EvaluatePredicate<PortGraph, NodeIndex> for PGPredicate {
    fn check(&self, bindings: &[impl Borrow<NodeIndex>], data: &PortGraph) -> bool {
        self.is_satisfied(data, &UnmanagedDenseMap::default(), bindings)
            .unwrap()
    }
}

impl<M, W> EvaluatePredicate<(&PortGraph, &M), NodeIndex> for PGPredicate<W>
where
    M: SecondaryMap<NodeIndex, W>,
    W: Clone + Ord + Debug,
{
    fn check(&self, bindings: &[impl Borrow<NodeIndex>], data: &(&PortGraph, &M)) -> bool {
        let &(graph, weights) = data;
        self.is_satisfied(graph, weights, bindings).unwrap()
    }
}

impl<W: std::fmt::Debug + Ord + Clone> ConditionalPredicate<PGIndexKey> for PGPredicate<W> {
    fn condition_on(
        &self,
        keys: &[PGIndexKey],
        known_constraints: &BTreeSet<Constraint<PGIndexKey, Self>>,
        _: &[Constraint<PGIndexKey, Self>],
    ) -> Satisfiable<Constraint<PGIndexKey, Self>> {
        let self_tags = self.get_tags(keys);
        // Only retain known constraints that have the same tag
        let known_constraints = known_constraints
            .iter()
            .filter(|c| c.get_tags().iter().any(|c| self_tags.contains(c)))
            .collect_vec();
        use PGPredicate::*;
        match self {
            HasNodeWeight(_) | IsConnected { .. } => {
                if known_constraints.is_empty() {
                    Satisfiable::Yes(self.clone().try_into_constraint(keys.to_vec()).unwrap())
                } else if known_constraints
                    .iter()
                    .all(|c| self == c.predicate() && c.required_bindings() == keys)
                {
                    Satisfiable::Tautology
                } else {
                    Satisfiable::No
                }
            }
            IsNotEqual { .. } => {
                let first_key = keys[0];
                let mut keys: BTreeSet<_> = keys[1..].iter().copied().collect();
                for s in known_constraints {
                    for k in s.required_bindings()[1..].iter() {
                        keys.remove(k);
                    }
                }
                if keys.is_empty() {
                    return Satisfiable::Tautology;
                }
                let mut args = vec![first_key];
                let n_other = keys.len();
                args.extend(keys);
                Satisfiable::Yes(
                    PGConstraint::try_new(PGPredicate::IsNotEqual { n_other }, args).unwrap(),
                )
            }
        }
    }
}

impl<W: Ord + Clone> ConstraintTag<PGIndexKey> for PGPredicate<W> {
    type Tag = PGTag;

    fn get_tags(&self, keys: &[PGIndexKey]) -> Vec<Self::Tag> {
        self.get_tags(keys)
    }
}

impl<W: Clone> Tag<PGIndexKey, PGPredicate<W>> for PGTag {
    type ExpansionFactor = u64;

    type Evaluator = DeterministicConstraintEvaluator<PGIndexKey, PGPredicate<W>>;

    fn expansion_factor<'c, C>(
        &self,
        constraints: impl IntoIterator<Item = C>,
    ) -> Self::ExpansionFactor
    where
        PGIndexKey: 'c,
        PGPredicate<W>: 'c,
        C: Into<(&'c PGPredicate<W>, &'c [PGIndexKey])>,
    {
        use PGTag::*;
        match self {
            // Deterministic so very cheap
            HasNodeWeight(..) => 1,
            // Deterministic so also very cheap (but prefer above)
            IsConnected(..) => 2,
            // As expensive as the number of constraints
            // (+ 2 so the others are always preferred)
            IsNotEqual(..) => 2 + constraints.into_iter().count() as u64,
        }
    }

    fn compile_evaluator<'c, C>(&self, constraints: impl IntoIterator<Item = C>) -> Self::Evaluator
    where
        PGPredicate<W>: 'c,
        C: Into<(&'c PGPredicate<W>, &'c [PGIndexKey])>,
    {
        let constraints = constraints.into_iter().map(|c| c.into()).collect_vec();
        Self::Evaluator::from_constraints(constraints)
    }
}

impl<W: Debug> Debug for PGPredicate<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PGPredicate::HasNodeWeight(weight) => write!(f, "has_node_weight({:?})", weight),
            PGPredicate::IsConnected {
                left_port,
                right_port,
                ..
            } => write!(f, "{:?} -> {:?}", left_port, right_port),
            PGPredicate::IsNotEqual { n_other } => write!(f, "not_equal({})", n_other),
        }
    }
}

fn has_edge<G: LinkView>(
    graph: &G,
    left: NodeIndex,
    left_port: PortOffset,
    right: NodeIndex,
    right_port: PortOffset,
) -> bool {
    let Some(left_port) = graph.port_index(left, left_port) else {
        return false;
    };
    graph
        .port_links(left_port)
        .any(|(_, p)| graph.port_offset(p) == Some(right_port) && graph.port_node(p) == Some(right))
}

#[cfg(test)]
mod tests {
    use portgraph::PortOffset;
    use rstest::fixture;

    use super::*;

    fn vname(root_id: usize, root_port: Option<PortOffset>, node_index: usize) -> PGIndexKey {
        if let Some(root_port) = root_port {
            PGIndexKey::AlongPath {
                path_root: root_id,
                path_start_port: root_port,
                path_length: node_index,
            }
        } else {
            assert_eq!(root_port, None);
            PGIndexKey::PathRoot { index: root_id }
        }
    }

    fn filter(
        left_port: PortOffset,
        (left_root_id, left_root_port, left_node_index): (usize, Option<PortOffset>, usize),
        right_port: PortOffset,
        (right_root_id, right_root_port, right_node_index): (usize, Option<PortOffset>, usize),
    ) -> PGConstraint {
        PGConstraint::try_new(
            PGPredicate::IsConnected {
                left_port,
                right_port,
            },
            vec![
                vname(left_root_id, left_root_port, left_node_index),
                vname(right_root_id, right_root_port, right_node_index),
            ],
        )
        .unwrap()
    }

    #[fixture]
    fn constraints() -> Vec<PGConstraint> {
        vec![
            filter(
                PortOffset::new_outgoing(0),
                (1, None, 0),
                PortOffset::new_incoming(0),
                (0, Some(PortOffset::new_incoming(0)), 1),
            ),
            filter(
                PortOffset::new_incoming(0),
                (2, None, 0),
                PortOffset::new_outgoing(0),
                (0, None, 0),
            ),
            filter(
                PortOffset::new_incoming(0),
                (0, Some(PortOffset::new_outgoing(0)), 2),
                PortOffset::new_outgoing(0),
                (0, Some(PortOffset::new_incoming(0)), 1),
            ),
            filter(
                PortOffset::new_incoming(0),
                (0, Some(PortOffset::new_outgoing(0)), 2),
                PortOffset::new_outgoing(1),
                (0, Some(PortOffset::new_incoming(0)), 1),
            ),
        ]
    }
}
