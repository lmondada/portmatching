use bimap::BiMap;
use derive_more::{From, Into};

use std::{
    hash::Hash,
    iter::{self, Map, Repeat, Zip},
    ops::RangeFrom,
};

use crate::{patterns::IterationStatus, Universe};

pub(crate) type SymbolsIter =
    Map<Zip<Repeat<IterationStatus>, RangeFrom<usize>>, fn((IterationStatus, usize)) -> Symbol>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, From, Into, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Symbol(IterationStatus, usize);

impl Symbol {
    pub(crate) fn new(status: IterationStatus, ind: usize) -> Self {
        Self(status, ind)
    }

    fn from_tuple((status, ind): (IterationStatus, usize)) -> Self {
        Self(status, ind)
    }

    pub(crate) fn root() -> Self {
        Self(IterationStatus::Skeleton(0), 0)
    }

    pub(crate) fn symbols_in_status(status: IterationStatus) -> SymbolsIter {
        iter::repeat(status).zip(0..).map(Self::from_tuple)
    }
}

/// Predicate to control allowable transitions
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum EdgePredicate<PNode, PEdge> {
    NodeProperty {
        node: Symbol,
        property: PNode,
    },
    LinkNewNode {
        node: Symbol,
        property: PEdge,
        new_node: Symbol,
    },
    LinkKnownNode {
        node: Symbol,
        property: PEdge,
        known_node: Symbol,
    },
    // Always true (non-deterministic)
    NextRoot {
        line_nb: usize,
        new_root: NodeLocation,
    },
    // Always true (non-deterministic)
    True,
    // Always true (deterministic)
    Fail,
}

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum NodeLocation {
    // The node is in an already-known location
    Exists(Symbol),
    // We need to explore along the i-th line to discover the node
    Discover(usize),
}

pub(crate) enum PredicateSatisfied<U> {
    NewSymbol(Symbol, U),
    Yes,
    No,
}

impl<PNode: Copy, PEdge: Copy> EdgePredicate<PNode, PEdge> {
    pub(crate) fn is_satisfied<'s, U: Universe>(
        &self,
        ass: &BiMap<Symbol, U>,
        node_prop: impl Fn(U, PNode) -> bool + 's,
        edge_prop: impl Fn(U, PEdge) -> Option<U> + 's,
    ) -> PredicateSatisfied<U> {
        match *self {
            EdgePredicate::NodeProperty { node, property } => {
                let u = *ass.get_by_left(&node).unwrap();
                if node_prop(u, property) {
                    PredicateSatisfied::Yes
                } else {
                    PredicateSatisfied::No
                }
            }
            EdgePredicate::LinkNewNode {
                node,
                property,
                new_node,
            } => {
                let u = *ass.get_by_left(&node).unwrap();
                let Some(new_u) = edge_prop(u, property) else {
                    return PredicateSatisfied::No;
                };
                if ass.get_by_right(&new_u).is_none() {
                    PredicateSatisfied::NewSymbol(new_node, new_u)
                } else {
                    PredicateSatisfied::No
                }
            }
            EdgePredicate::LinkKnownNode {
                node,
                property,
                known_node,
            } => {
                let u = *ass.get_by_left(&node).unwrap();
                let Some(new_u) = edge_prop(u, property) else {
                    return PredicateSatisfied::No;
                };
                if ass.get_by_left(&known_node).unwrap() == &new_u {
                    PredicateSatisfied::Yes
                } else {
                    PredicateSatisfied::No
                }
            }
            EdgePredicate::True { .. } | EdgePredicate::NextRoot { .. } | EdgePredicate::Fail => {
                PredicateSatisfied::Yes
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PredicateCompatibility {
    Deterministic,
    NonDeterministic,
    Incompatible,
}

/// Partition of edge predicates into compatible equivalence classes
///
/// Any predicate belongs to one of the following equivalence classes.
/// All predicates within a class are compatible with eachother.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CompatibilityType {
    NonDetType,
    LinkType(Symbol),
    WeightType(Symbol),
    FailType,
}

impl CompatibilityType {
    fn transition_type(&self) -> PredicateCompatibility {
        match self {
            Self::NonDetType => PredicateCompatibility::NonDeterministic,
            Self::LinkType(_) => PredicateCompatibility::Deterministic,
            Self::WeightType(_) => PredicateCompatibility::Deterministic,
            Self::FailType => PredicateCompatibility::Deterministic,
        }
    }

    fn from_predicate<PNode, PEdge>(pred: &EdgePredicate<PNode, PEdge>) -> Self {
        match pred {
            EdgePredicate::True | EdgePredicate::NextRoot { .. } => Self::NonDetType,
            EdgePredicate::Fail => Self::FailType,
            EdgePredicate::LinkNewNode { node, .. } | EdgePredicate::LinkKnownNode { node, .. } => {
                Self::LinkType(*node)
            }
            EdgePredicate::NodeProperty { node, .. } => Self::WeightType(*node),
        }
    }

    fn is_compatible(&self, other: CompatibilityType) -> bool {
        if other == Self::FailType && matches!(self, Self::LinkType(_) | Self::WeightType(_)) {
            true
        } else if self == &Self::FailType
            && matches!(other, Self::LinkType(_) | Self::WeightType(_))
        {
            true
        } else {
            self == &other
        }
    }
}

pub(crate) fn are_compatible_predicates<'a, PNode: 'a, PEdge: 'a>(
    preds: impl IntoIterator<Item = &'a EdgePredicate<PNode, PEdge>>,
) -> PredicateCompatibility {
    let mut preds = preds.into_iter().map(CompatibilityType::from_predicate);
    let Some(first) = preds.next() else {
        return PredicateCompatibility::Deterministic;
    };
    if preds.all(|c| c.is_compatible(first)) {
        first.transition_type()
    } else {
        PredicateCompatibility::Incompatible
    }
}
