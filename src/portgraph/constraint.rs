//! Constraints for port graph matching.
//!
//! There are two main logic bits
//!  - how to express several constraints as mutually exclusive
//!    constraints; this can be found in the mutex submodule.
//!  - how to decompose a constraint into a list of constraints. Note that this
//!    has to be consistent with how the indexing scheme binds key-values.

mod mutex;

use std::{cmp, collections::BTreeSet, iter};

use itertools::Itertools;
use petgraph::visit::EdgeCount;
use portgraph::{NodeIndex, PortGraph, PortView};

use crate::{
    constraint::DetHeuristic,
    mutex_tree::{ConditionedPredicate, MutuallyExclusiveTree, ToConstraintsTree},
    utils::{portgraph::line_partition, sort_with_indices},
    Constraint, HashMap,
};

use super::{indexing::PGIndexKey, predicate::PGPredicate};
use mutex::*;

/// A constraint on a port graph.
pub type PGConstraint<W = ()> = Constraint<PGIndexKey, PGPredicate<W>>;

impl ToConstraintsTree<PGIndexKey> for PGPredicate {
    fn to_constraints_tree(constraints: Vec<PGConstraint>) -> MutuallyExclusiveTree<PGConstraint> {
        if constraints.is_empty() {
            return MutuallyExclusiveTree::new();
        }
        let constraints = sort_with_indices(constraints);
        // This will always add the first constraint to the tree, plus any other
        // that are mutually exclusive
        mutex_filter(constraints)
    }
}

impl ConditionedPredicate<PGIndexKey> for PGPredicate {
    fn conditioned(
        constraint: &Constraint<PGIndexKey, Self>,
        satisfied: &[&Constraint<PGIndexKey, Self>],
    ) -> Option<Constraint<PGIndexKey, Self>> {
        if !matches!(constraint.predicate(), PGPredicate::IsNotEqual { .. }) {
            return Some(constraint.clone());
        }
        let first_key = constraint.required_bindings()[0];
        let mut keys: BTreeSet<_> = constraint.required_bindings()[1..]
            .iter()
            .copied()
            .collect();
        for s in satisfied
            .iter()
            .filter(|s| s.required_bindings()[0] == first_key)
        {
            for k in s.required_bindings()[1..].iter().copied() {
                keys.remove(&k);
            }
        }
        if keys.is_empty() {
            return None;
        }
        let mut args = vec![first_key];
        let n_other = keys.len();
        args.extend(keys);
        Some(PGConstraint::try_new(PGPredicate::IsNotEqual { n_other }, args).unwrap())
    }
}

impl DetHeuristic for PGConstraint {
    fn make_det(_: &[&Self]) -> bool {
        true
    }
}

impl PartialOrd for PGConstraint {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PGConstraint {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.max_key().cmp(&other.max_key())
    }
}

impl PGConstraint {
    /// The key to use for comparisons.
    ///
    /// Constraints are sorted by their largest key binding.
    fn max_key(&self) -> (PGIndexKey, &PGPredicate) {
        let max_key = self
            .required_bindings()
            .iter()
            .max()
            .copied()
            .unwrap_or(PGIndexKey::PathRoot { index: 0 });
        (max_key, self.predicate())
    }
}

pub(super) fn constraint_vec(graph: &PortGraph, root: NodeIndex) -> Vec<PGConstraint> {
    if graph.edge_count() == 0 {
        return vec![PGConstraint::try_new(
            PGPredicate::HasNodeWeight(()),
            vec![PGIndexKey::root(0)],
        )
        .unwrap()];
    }
    let mut constraints = Vec::new();
    let mut node_to_key = HashMap::from_iter([(root, PGIndexKey::root(0))]);
    let mut node_to_root_ind = HashMap::from_iter([(root, 0)]);
    for line in line_partition(&graph, root) {
        let root = graph.port_node(line[0].0).unwrap();
        let n_roots = node_to_root_ind.len();
        let root_index = *node_to_root_ind.entry(root).or_insert(n_roots);
        let root_offset = graph.port_offset(line[0].0).unwrap();
        for (i, (left, right)) in line.into_iter().enumerate() {
            let left_node = graph.port_node(left).unwrap();
            let left_port = graph.port_offset(left).unwrap();
            let right_node = graph.port_node(right).unwrap();
            let right_port = graph.port_offset(right).unwrap();
            let left_key = *node_to_key.get(&left_node).expect("unknown edge LHS");
            let right_key = if let Some(right_key) = node_to_key.get(&right_node) {
                *right_key
            } else {
                // Create a new key
                let key = PGIndexKey::AlongPath {
                    path_root: root_index,
                    path_start_port: root_offset,
                    path_length: i + 1,
                };
                // Ensure it does not clash with previously bound keys
                let args = iter::once(key)
                    .chain(node_to_key.values().copied())
                    .collect_vec();
                constraints.push(
                    PGConstraint::try_new(
                        PGPredicate::IsNotEqual {
                            n_other: node_to_key.len(),
                        },
                        args,
                    )
                    .unwrap(),
                );
                node_to_key.insert(right_node, key);
                key
            };
            constraints.push(
                PGConstraint::try_new(
                    PGPredicate::IsConnected {
                        left_port,
                        right_port,
                    }
                    .into(),
                    vec![left_key, right_key],
                )
                .unwrap(),
            );
        }
    }
    if constraints.is_empty() {
        // We add one (dummy) constraint for the empty pattern, forcing
        // the matcher to bind the first character to a position in the
        // string when matched. An alternative would be to explicitly
        // disallow empty patterns.
        constraints.push(
            PGConstraint::try_new(
                PGPredicate::IsNotEqual { n_other: 0 },
                vec![PGIndexKey::root(0)],
            )
            .unwrap(),
        );
    }
    constraints
}

#[cfg(test)]
mod tests {
    use portgraph::PortOffset;
    use rstest::{fixture, rstest};

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
            }
            .into(),
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

    #[rstest]
    fn test_to_mut_ex_assign_0(constraints: Vec<PGConstraint>) {
        let tree = PGPredicate::to_constraints_tree(constraints.clone());
        assert_eq!(
            tree.children(tree.root())
                .map(|(_, c)| c.clone())
                .collect_vec(),
            [
                filter(
                    PortOffset::new_incoming(0),
                    (0, Some(PortOffset::new_outgoing(0)), 2),
                    PortOffset::new_outgoing(0),
                    (0, Some(PortOffset::new_incoming(0)), 1)
                ),
                filter(
                    PortOffset::new_incoming(0),
                    (0, Some(PortOffset::new_outgoing(0)), 2),
                    PortOffset::new_outgoing(1),
                    (0, Some(PortOffset::new_incoming(0)), 1),
                ),
            ]
        );
    }

    #[rstest]
    fn test_to_mut_ex_assign_1(constraints: Vec<PGConstraint>) {
        let tree = PGPredicate::to_constraints_tree(constraints[..2].to_vec());
        assert_eq!(
            tree.children(tree.root())
                .map(|(_, c)| c.clone())
                .collect_vec(),
            [filter(
                PortOffset::new_outgoing(0),
                (1, None, 0),
                PortOffset::new_incoming(0),
                (0, Some(PortOffset::new_incoming(0)), 1),
            ),]
        );
    }

    #[test]
    fn test_constraint_ordering() {
        let a = filter(
            PortOffset::new_incoming(0),
            (0, Some(PortOffset::new_outgoing(0)), 1),
            PortOffset::new_outgoing(0),
            (0, None, 0),
        );
        let b = filter(
            PortOffset::new_incoming(0),
            (0, Some(PortOffset::new_outgoing(0)), 1),
            PortOffset::new_outgoing(0),
            (1, None, 0),
        );
        // For same literal, an AssignPredicate should be smaller
        assert!(a < b);
    }
}
