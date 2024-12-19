//! Predicates (the core of constraints), for port graph matching.

use std::{borrow::Borrow, fmt::Debug};

use itertools::Itertools;
use portgraph::{LinkView, NodeIndex, PortGraph, PortOffset, SecondaryMap, UnmanagedDenseMap};

use crate::predicate::{ArityPredicate, Predicate};

/// A predicate for constraints on (weighted) port graph
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PGPredicate<NodeWeight = ()> {
    /// The node has a specific weight
    HasNodeWeight(NodeWeight),
    /// The two nodes are connected by an edge from `left_port` to `right_port`
    IsConnected {
        /// The left port of the edge
        left_port: PortOffset,
        /// The right port of the edge
        right_port: PortOffset,
    },
    /// The node is not equal to any of the other `n_other` nodes
    IsNotEqual {
        /// The number of other nodes, determining the predicate arity.
        n_other: usize,
    },
}

impl<W: Eq + Debug> PGPredicate<W> {
    /// Whether the predicate is satisfied, given graph, node weights and arguments
    fn is_satisfied(
        &self,
        graph: &PortGraph,
        node_weights: &impl SecondaryMap<NodeIndex, W>,
        args: &[impl Borrow<NodeIndex>],
    ) -> Result<bool, String> {
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
}

impl<W: Clone + Eq> ArityPredicate for PGPredicate<W> {
    fn arity(&self) -> usize {
        match self {
            PGPredicate::HasNodeWeight(..) => 1,
            PGPredicate::IsConnected { .. } => 2,
            PGPredicate::IsNotEqual { n_other } => n_other + 1,
        }
    }
}

impl Predicate<PortGraph> for PGPredicate {
    type InvalidPredicateError = String;

    fn check(&self, data: &PortGraph, args: &[impl Borrow<NodeIndex>]) -> Result<bool, String> {
        self.is_satisfied(data, &UnmanagedDenseMap::default(), args)
    }
}

impl<M, W> Predicate<(&PortGraph, &M)> for PGPredicate<W>
where
    M: SecondaryMap<NodeIndex, W>,
    W: Clone + Eq + Debug,
{
    type InvalidPredicateError = String;

    fn check(
        &self,
        data: &(&PortGraph, &M),
        args: &[impl Borrow<NodeIndex>],
    ) -> Result<bool, String> {
        let &(graph, weights) = data;
        self.is_satisfied(graph, weights, args)
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
        .filter(|&(_, p)| {
            graph.port_offset(p) == Some(right_port) && graph.port_node(p) == Some(right)
        })
        .next()
        .is_some()
}
