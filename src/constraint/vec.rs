use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
};

use portgraph::{NodeIndex, PortGraph, PortIndex};

use crate::utils::ZeroRange;

use super::{ElementaryConstraint, NodeRange};

/// A vector of constraints, with simplifying logic.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintVec<N> {
    Vec(Vec<ElementaryConstraint<N>>),
    Impossible,
}

impl<N> Default for ConstraintVec<N> {
    fn default() -> Self {
        ConstraintVec::Vec(vec![])
    }
}

impl<N: Clone + Eq> FromIterator<ElementaryConstraint<N>> for ConstraintVec<N> {
    fn from_iter<T: IntoIterator<Item = ElementaryConstraint<N>>>(iter: T) -> Self {
        ConstraintVec::new(iter.into_iter().collect())
    }
}

impl<N: Clone + Eq> ConstraintVec<N> {
    pub(super) fn new(constraints: Vec<ElementaryConstraint<N>>) -> Self {
        let mut constraints = ConstraintVec::Vec(constraints);
        constraints.simplify();
        constraints
    }

    pub(super) fn is_satisfied(
        &self,
        port: PortIndex,
        graph: &PortGraph,
        root: NodeIndex,
        weight: &N,
    ) -> bool
    where
        N: Eq,
    {
        let ConstraintVec::Vec(constraints) = self else { return false };
        constraints
            .iter()
            .all(|c| c.is_satisfied(port, graph, root, weight))
    }

    pub(super) fn is_impossible(&self) -> bool {
        matches!(self, ConstraintVec::Impossible)
    }

    pub(super) fn and(&self, other: &Self) -> Option<Self>
    where
        N: Clone,
    {
        let c1 = match self {
            ConstraintVec::Vec(v) => v,
            ConstraintVec::Impossible => return None,
        };
        let c2 = match other {
            ConstraintVec::Vec(v) => v,
            ConstraintVec::Impossible => return None,
        };
        let c: Self = c1.iter().cloned().chain(c2.iter().cloned()).collect();
        if c.is_impossible() {
            return None;
        }
        Some(c)
    }

    fn simplify(&mut self)
    where
        N: Eq,
    {
        let ConstraintVec::Vec(constraints) = self else { return };

        let mut addresses = BTreeSet::new();
        let mut merged_no_addresses = BTreeMap::new();
        let mut all_labels = Vec::new();
        let mut all_weights = Vec::new();
        for c in mem::take(constraints) {
            match c {
                ElementaryConstraint::PortLabel(l) => all_labels.push(l),
                ElementaryConstraint::NodeWeight(w) => all_weights.push(w),
                ElementaryConstraint::NoMatch(NodeRange { spine, range }) => {
                    merged_no_addresses
                        .entry(spine)
                        .and_modify(|curr: &mut ZeroRange| curr.merge(&range))
                        .or_insert(range.clone());
                }
                ElementaryConstraint::Match(addr) => {
                    addresses.insert(addr);
                }
            }
        }

        let mut label = all_labels.first().cloned();
        for l in all_labels {
            label = label.and_then(|label| label.and(l));
            if label.is_none() {
                *self = Self::Impossible;
                return;
            }
        }
        let weight = all_weights.first().cloned();
        if weight.is_some()
            && all_weights
                .into_iter()
                .any(|w| &w != weight.as_ref().unwrap())
        {
            *self = Self::Impossible;
            return;
        };

        let no_addresses = merged_no_addresses
            .into_iter()
            .map(|(spine, range)| NodeRange { spine, range })
            .collect::<Vec<_>>();

        for address in addresses.iter() {
            if no_addresses.iter().any(|range| range.contains(address)) {
                *self = Self::Impossible;
                return;
            }
        }

        *constraints = weight
            .map(ElementaryConstraint::NodeWeight)
            .into_iter()
            .chain(label.map(ElementaryConstraint::PortLabel))
            .chain(addresses.into_iter().map(ElementaryConstraint::Match))
            .chain(no_addresses.into_iter().map(ElementaryConstraint::NoMatch))
            .collect();
    }
}
