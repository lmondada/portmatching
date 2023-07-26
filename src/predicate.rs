use derive_more::{From, Into};

use std::{
    hash::Hash,
    iter::{self, Map, Repeat, Zip},
    ops::RangeFrom,
};

use crate::{patterns::IterationStatus, BiMap, EdgeProperty, NodeProperty, Universe};

pub(crate) type SymbolsIter =
    Map<Zip<Repeat<IterationStatus>, RangeFrom<usize>>, fn((IterationStatus, usize)) -> Symbol>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Symbol(pub(crate) IterationStatus, pub(crate) usize);

impl Symbol {
    pub(crate) fn new(status: IterationStatus, ind: usize) -> Self {
        Self(status, ind)
    }

    fn from_tuple((status, ind): (IterationStatus, usize)) -> Self {
        Self::new(status, ind)
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
pub(crate) enum EdgePredicate<PNode, PEdge, OffsetID> {
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
        offset: OffsetID,
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

impl<PNode: NodeProperty, PEdge: EdgeProperty> EdgePredicate<PNode, PEdge, PEdge::OffsetID> {
    pub(crate) fn is_satisfied<'s, U: Universe>(
        &self,
        ass: &BiMap<Symbol, U>,
        node_prop: impl for<'a> Fn(U, &'a PNode) -> bool + 's,
        edge_prop: impl for<'a> Fn(U, &'a PEdge) -> Option<U> + 's,
    ) -> PredicateSatisfied<U> {
        match self {
            EdgePredicate::NodeProperty { node, property } => {
                let u = *ass.get_by_left(node).unwrap();
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
                let u = *ass.get_by_left(node).unwrap();
                let Some(new_u) = edge_prop(u, property) else {
                    return PredicateSatisfied::No;
                };
                if ass.get_by_right(&new_u).is_none() {
                    PredicateSatisfied::NewSymbol(*new_node, new_u)
                } else {
                    PredicateSatisfied::No
                }
            }
            EdgePredicate::LinkKnownNode {
                node,
                property,
                known_node,
            } => {
                let u = *ass.get_by_left(node).unwrap();
                let Some(new_u) = edge_prop(u, property) else {
                    return PredicateSatisfied::No;
                };
                if ass.get_by_left(known_node).unwrap() == &new_u {
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

    pub(crate) fn transition_type(&self) -> PredicateCompatibility
    where
        PEdge: EdgeProperty,
    {
        CompatibilityType::from_predicate(self).transition_type()
    }

    pub(crate) fn is_compatible(&self, other: &Self) -> bool
    where
        PEdge: EdgeProperty,
    {
        let c1 = CompatibilityType::from_predicate(self);
        let c2 = CompatibilityType::from_predicate(other);
        c1.is_compatible(c2)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
enum CompatibilityType<OffsetID> {
    NonDet,
    Link(Symbol, OffsetID),
    Weight(Symbol),
    Fail,
}

impl<OffsetID> CompatibilityType<OffsetID> {
    fn transition_type(&self) -> PredicateCompatibility {
        match self {
            Self::NonDet => PredicateCompatibility::NonDeterministic,
            Self::Link(_, _) => PredicateCompatibility::Deterministic,
            Self::Weight(_) => PredicateCompatibility::Deterministic,
            Self::Fail => PredicateCompatibility::Deterministic,
        }
    }

    fn from_predicate<PNode, PEdge>(pred: &EdgePredicate<PNode, PEdge, PEdge::OffsetID>) -> Self
    where
        PEdge: EdgeProperty<OffsetID = OffsetID>,
    {
        match pred {
            EdgePredicate::True | EdgePredicate::NextRoot { .. } => Self::NonDet,
            EdgePredicate::Fail => Self::Fail,
            EdgePredicate::LinkNewNode { node, property, .. }
            | EdgePredicate::LinkKnownNode { node, property, .. } => {
                Self::Link(*node, property.offset_id())
            }
            EdgePredicate::NodeProperty { node, .. } => Self::Weight(*node),
        }
    }

    fn is_compatible(&self, other: CompatibilityType<OffsetID>) -> bool
    where
        OffsetID: Eq,
    {
        if other == Self::Fail && matches!(self, Self::Link(_, _) | Self::Weight(_))
            || self == &Self::Fail && matches!(other, Self::Link(_, _) | Self::Weight(_))
        {
            true
        } else {
            self == &other
        }
    }
}

pub(crate) fn are_compatible_predicates<'a, PNode, PEdge>(
    preds: impl IntoIterator<Item = &'a EdgePredicate<PNode, PEdge, PEdge::OffsetID>>,
) -> PredicateCompatibility
where
    PNode: NodeProperty + 'a,
    PEdge: EdgeProperty + 'a,
{
    let mut preds = preds.into_iter();
    let Some(first) = preds.next() else {
        return PredicateCompatibility::Deterministic;
    };
    if preds.all(|c| c.is_compatible(first)) {
        first.transition_type()
    } else {
        PredicateCompatibility::Incompatible
    }
}
