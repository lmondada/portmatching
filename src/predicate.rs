use bimap::BiMap;

use std::hash::Hash;

use crate::Universe;

/// Predicate to control allowable transitions
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) enum EdgePredicate<PNode, PEdge, Symbol> {
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
    True {
        line: usize,
        deterministic: bool,
    },
}

pub(crate) enum PredicateSatisfied<U, Symbol> {
    NewSymbol(Symbol, U),
    Yes,
    No,
}

impl<PNode: Copy, PEdge: Copy, Symbol: Copy + Eq + Hash> EdgePredicate<PNode, PEdge, Symbol> {
    pub(crate) fn is_satisfied<'s, U: Universe>(
        &self,
        ass: &BiMap<Symbol, U>,
        node_prop: impl Fn(U, PNode) -> bool + 's,
        edge_prop: impl Fn(U, PEdge) -> Option<U> + 's,
    ) -> PredicateSatisfied<U, Symbol> {
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
            EdgePredicate::True { .. } => PredicateSatisfied::Yes,
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
enum CompatibilityType<Symbol> {
    NonDetType,
    LinkType(Symbol),
    WeightType(Symbol),
    FailType,
}

impl<Symbol: Eq + Copy> CompatibilityType<Symbol> {
    fn transition_type(&self) -> PredicateCompatibility {
        match self {
            Self::NonDetType => PredicateCompatibility::NonDeterministic,
            Self::LinkType(_) => PredicateCompatibility::Deterministic,
            Self::WeightType(_) => PredicateCompatibility::Deterministic,
            Self::FailType => PredicateCompatibility::Deterministic,
        }
    }

    fn from_predicate<PNode, PEdge>(pred: &EdgePredicate<PNode, PEdge, Symbol>) -> Self {
        match pred {
            EdgePredicate::True {
                deterministic: false,
                ..
            } => Self::NonDetType,
            EdgePredicate::True {
                deterministic: true,
                ..
            } => Self::FailType,
            EdgePredicate::LinkNewNode { node, .. } | EdgePredicate::LinkKnownNode { node, .. } => {
                Self::LinkType(*node)
            }
            EdgePredicate::NodeProperty { node, .. } => Self::WeightType(*node),
        }
    }

    fn is_compatible(&self, other: CompatibilityType<Symbol>) -> bool {
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

pub(crate) fn are_compatible_predicates<'a, PNode: 'a, PEdge: 'a, Symbol: Copy + Eq + 'a>(
    preds: impl IntoIterator<Item = &'a EdgePredicate<PNode, PEdge, Symbol>>,
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
