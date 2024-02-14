use std::hash::Hash;

use delegate::delegate;
use derive_more::{Into, From};
use portgraph::{NodeIndex, PortGraph, PortOffset, Weights};

use crate::{
    constraint::{ConstraintSplit, ScopeConstraint},
    BiMap,
};

use super::{
    pattern::UnweightedEdge, EdgeProperty, NodeLocation, NodeProperty, Symbol, WeightedPortGraphRef,
};

/// Predicate to control allowable transitions
#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PortgraphConstraint<PNode, PEdge, OffsetID> {
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

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug, Into, From)]
pub struct UnweightedPortgraphConstraint<OffsetID>(
    PortgraphConstraint<(), UnweightedEdge, OffsetID>,
);

impl<PNode: NodeProperty +'static, PEdge: EdgeProperty + 'static> ScopeConstraint
    for PortgraphConstraint<PNode, PEdge, PEdge::OffsetID>
{
    type Symbol = Symbol;
    type Value = NodeIndex;
    type DataRef<'a> = WeightedPortGraphRef<'a, Weights<PNode, PEdge>>;

    fn scope(&self) -> crate::constraint::Scope<Self> {
        todo!()
    }

    fn new_symbols(&self) -> crate::constraint::Scope<Self> {
        todo!()
    }

    fn is_satisfied<'a>(
        &self,
        input: Self::DataRef<'a>,
        scope: &crate::constraint::ScopeMap<Self>,
    ) -> Option<crate::constraint::ScopeMap<Self>> {
        todo!()
    }

    fn split<'a>(constraints: impl Iterator<Item = &'a Self>) -> ConstraintSplit<'a, Self> {
        todo!()
    }

    fn uid(&self) -> Option<String> {
        todo!()
    }
}

impl ScopeConstraint for UnweightedPortgraphConstraint<PortOffset> {
    type Symbol = Symbol;
    type Value = NodeIndex;
    type DataRef<'d> = &'d PortGraph;

    fn scope(&self) -> crate::constraint::Scope<Self> {
        self.0.scope()
    }

    fn new_symbols(&self) -> crate::constraint::Scope<Self> {
        self.0.new_symbols()
    }

    fn is_satisfied(
        &self,
        input: Self::DataRef<'_>,
        scope: &crate::constraint::ScopeMap<Self>,
    ) -> Option<crate::constraint::ScopeMap<Self>> {
        self.0.is_satisfied((input, &Weights::new()).into(), scope)
    }

    fn split<'a>(constraints: impl Iterator<Item = &'a Self>) -> ConstraintSplit<'a, Self> {
        todo!()
    }

    fn uid(&self) -> Option<String> {
        self.0.uid()
    }
}

pub(crate) enum PredicateSatisfied<U> {
    NewSymbol(Symbol, U),
    Yes,
    No,
}

// impl<PNode: NodeProperty, PEdge: EdgeProperty> PortgraphConstraint<PNode, PEdge, PEdge::OffsetID> {
//     pub(crate) fn is_satisfied<'s, U: Eq + Hash + Copy>(
//         &self,
//         ass: &BiMap<Symbol, U>,
//         node_prop: impl for<'a> Fn(U, &'a PNode) -> bool + 's,
//         edge_prop: impl for<'a> Fn(U, &'a PEdge) -> Option<U> + 's,
//     ) -> PredicateSatisfied<U> {
//         match self {
//             PortgraphConstraint::NodeProperty { node, property } => {
//                 let u = *ass.get_by_left(node).unwrap();
//                 if node_prop(u, property) {
//                     PredicateSatisfied::Yes
//                 } else {
//                     PredicateSatisfied::No
//                 }
//             }
//             PortgraphConstraint::LinkNewNode {
//                 node,
//                 property,
//                 new_node,
//             } => {
//                 let u = *ass.get_by_left(node).unwrap();
//                 let Some(new_u) = edge_prop(u, property) else {
//                     return PredicateSatisfied::No;
//                 };
//                 if ass.get_by_right(&new_u).is_none() {
//                     PredicateSatisfied::NewSymbol(*new_node, new_u)
//                 } else {
//                     PredicateSatisfied::No
//                 }
//             }
//             PortgraphConstraint::LinkKnownNode {
//                 node,
//                 property,
//                 known_node,
//             } => {
//                 let u = *ass.get_by_left(node).unwrap();
//                 let Some(new_u) = edge_prop(u, property) else {
//                     return PredicateSatisfied::No;
//                 };
//                 if ass.get_by_left(known_node).unwrap() == &new_u {
//                     PredicateSatisfied::Yes
//                 } else {
//                     PredicateSatisfied::No
//                 }
//             }
//             PortgraphConstraint::True { .. }
//             | PortgraphConstraint::NextRoot { .. }
//             | PortgraphConstraint::Fail => PredicateSatisfied::Yes,
//         }
//     }

//     pub(crate) fn transition_type(&self) -> PredicateCompatibility
//     where
//         PEdge: EdgeProperty,
//     {
//         CompatibilityType::from_predicate(self).transition_type()
//     }

//     pub(crate) fn is_compatible(&self, other: &Self) -> bool
//     where
//         PEdge: EdgeProperty,
//     {
//         let c1 = CompatibilityType::from_predicate(self);
//         let c2 = CompatibilityType::from_predicate(other);
//         c1.is_compatible(c2)
//     }
// }

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

    fn from_predicate<PNode, PEdge>(
        pred: &PortgraphConstraint<PNode, PEdge, PEdge::OffsetID>,
    ) -> Self
    where
        PEdge: EdgeProperty<OffsetID = OffsetID>,
    {
        match pred {
            PortgraphConstraint::True | PortgraphConstraint::NextRoot { .. } => Self::NonDet,
            PortgraphConstraint::Fail => Self::Fail,
            PortgraphConstraint::LinkNewNode { node, property, .. }
            | PortgraphConstraint::LinkKnownNode { node, property, .. } => {
                Self::Link(*node, property.offset_id())
            }
            PortgraphConstraint::NodeProperty { node, .. } => Self::Weight(*node),
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

// pub(crate) fn are_compatible_predicates<'a, PNode, PEdge>(
//     preds: impl IntoIterator<Item = &'a PortgraphConstraint<PNode, PEdge, PEdge::OffsetID>>,
// ) -> PredicateCompatibility
// where
//     PNode: NodeProperty + 'a,
//     PEdge: EdgeProperty + 'a,
// {
//     let mut preds = preds.into_iter();
//     let Some(first) = preds.next() else {
//         return PredicateCompatibility::Deterministic;
//     };
//     if preds.all(|c| c.is_compatible(first)) {
//         first.transition_type()
//     } else {
//         PredicateCompatibility::Incompatible
//     }
// }
