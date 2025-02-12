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
use thiserror::Error;

use crate::{
    utils::{portgraph::line_partition, sort_with_indices},
    Constraint, HashMap,
};

use super::{indexing::PGIndexKey, predicate::PGPredicate};
use mutex::*;

/// A constraint on a port graph.
pub type PGConstraint<W = ()> = Constraint<PGIndexKey, PGPredicate<W>>;

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
