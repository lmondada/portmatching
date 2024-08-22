//! Logic to compute mutually exclusive PGConstraints.

use std::collections::VecDeque;

use itertools::Itertools;

use crate::{mutex_tree::MutuallyExclusiveTree, portgraph::predicate::PGPredicate};

use super::PGConstraint;

/// Build a constraint tree for filter constraints.
///
/// Strategy: take the smallest filter (the first), and add all the other filters
/// that are mutually exclusive with it. Filters are mutually exclusive if they
/// are of the same enum variant and
///  - for PGPredicate::IsConnected: they share the same left key and left_port.
///  - for PGPredicate::HasNodeWeight: they share the same key.
///  - for PGPredicate::IsNotEqual: they match on the first key, and if A resp. B
///    are the set of remaining keys (i.e. all but the first argument), Then we
///    create three IsNotEqual constraints, checking for clash with (i) A ∧ B,
///    (ii) A ∧ ¬B and (iii) ¬A ∧ B.
///    We use the fact that constraints are checked in order to simplify this
///    to the equivalent sequence of constraints (i) A ∧ B, (ii) A, (iii) B.
///
/// This assumes that all constraints are distinct and the vec is not empty.
pub(super) fn mutex_filter(
    constraints: Vec<(PGConstraint, usize)>,
) -> MutuallyExclusiveTree<PGConstraint> {
    // Assume constraints is not empty
    let first_constraint = &constraints[0].0;

    match first_constraint.predicate() {
        PGPredicate::HasNodeWeight(..) => mutex_has_node_weight(constraints),
        PGPredicate::IsConnected { .. } => mutex_is_connected(constraints),
        PGPredicate::IsNotEqual { .. } => mutex_is_not_equal(constraints),
    }
}

fn mutex_has_node_weight<W>(
    constraints: Vec<(PGConstraint<W>, usize)>,
) -> MutuallyExclusiveTree<PGConstraint<W>> {
    let key_0 = constraints[0].0.required_bindings()[0];
    let is_weight_pred =
        |c: &PGConstraint<W>| matches!(c.predicate(), PGPredicate::HasNodeWeight(_));
    let key_eq = |c: &PGConstraint<W>| c.required_bindings()[0] == key_0;
    let constraints = constraints
        .into_iter()
        .filter(|(c, _)| is_weight_pred(c) && key_eq(c))
        .map(|(c, i)| (c, vec![i]));
    MutuallyExclusiveTree::with_children(constraints)
}

fn mutex_is_connected<W>(
    constraints: Vec<(PGConstraint<W>, usize)>,
) -> MutuallyExclusiveTree<PGConstraint<W>> {
    let first_key_0 = constraints[0].0.required_bindings()[0];
    let left_port_0 = match &constraints[0].0.predicate() {
        PGPredicate::IsConnected { left_port, .. } => *left_port,
        _ => panic!("expected IsConnected",),
    };
    let is_link_pred = |c: &PGConstraint<W>| {
        let &PGPredicate::IsConnected { left_port, .. } = c.predicate() else {
            return false;
        };
        left_port == left_port_0
    };
    let first_key_eq = |c: &PGConstraint<W>| c.required_bindings()[0] == first_key_0;
    let constraints = constraints
        .into_iter()
        .filter(|(c, _)| is_link_pred(c) && first_key_eq(c))
        .map(|(c, i)| (c, vec![i]));
    MutuallyExclusiveTree::with_children(constraints)
}

pub(super) fn mutex_is_not_equal<W: Clone + Eq>(
    constraints: Vec<(PGConstraint<W>, usize)>,
) -> MutuallyExclusiveTree<PGConstraint<W>> {
    // Assume constraints is not empty
    let first_constraint = &constraints[0].0;
    let first_key_0 = first_constraint.required_bindings()[0];

    // We can only turn IsNotEqual constraints into mutually exclusive predicates
    // if they act on the same variable
    let is_not_equal_pred =
        |c: &PGConstraint<W>| matches!(c.predicate(), PGPredicate::IsNotEqual { .. });
    let first_key_eq = |c: &PGConstraint<W>| c.required_bindings()[0] == first_key_0;
    let constraints = constraints
        .into_iter()
        .filter(|(c, _)| is_not_equal_pred(c) && first_key_eq(c))
        .collect_vec();

    all_combinations(&constraints)
}

/// Combines assigns with identical assignment and ports in all possible ways.
///
/// This has a combinatorial blow-up in `assigns` size, but in practice, cases
/// where this gets big should be rare.
fn all_combinations<W: Clone + Eq>(
    constraints: &[(PGConstraint<W>, usize)],
) -> MutuallyExclusiveTree<PGConstraint<W>> {
    if constraints.is_empty() {
        return MutuallyExclusiveTree::new();
    }
    let first_key = constraints[0].0.required_bindings()[0];

    // For each constraint collect the keys to be checked against
    let key_sets = constraints
        .iter()
        .map(|(c, _)| &c.required_bindings()[1..])
        .collect_vec();
    let constraints_indices = constraints.iter().map(|(_, i)| *i).collect_vec();

    // Start by putting the constraints themselves. We will put combinations
    // of these constraints as descendants
    let mut tree = MutuallyExclusiveTree::with_children(
        constraints
            .iter()
            .map(|(c, i)| (c.clone(), vec![*i]))
            .collect_vec(),
    );

    // A queue of nodes in the tree to process, stored as
    // (largest_constraint_index, node_index, keys_checked)
    let mut queue: VecDeque<_> = tree
        .children(tree.root())
        .into_iter()
        .enumerate()
        .map(|(i, (child_ind, _))| (i, child_ind, key_sets[i].to_vec()))
        .collect();

    while let Some((largest_ind, tree_node, key_set)) = queue.pop_front() {
        for new_largest_ind in (largest_ind + 1)..constraints.len() {
            let mut new_key_set = key_sets[new_largest_ind].to_vec();
            new_key_set.retain(|&k| !key_set.contains(&k));
            if !new_key_set.is_empty() {
                let mut args = vec![first_key];
                args.extend(new_key_set.clone());
                let new_constraint = PGConstraint::try_new(
                    PGPredicate::IsNotEqual {
                        n_other: new_key_set.len(),
                    },
                    args,
                )
                .unwrap();
                let new_child = tree.add_child(tree_node, new_constraint);
                tree.add_constraint_index(new_child, constraints_indices[new_largest_ind])
                    .unwrap();
                queue.push_back((new_largest_ind, new_child, new_key_set));
            } else {
                tree.add_constraint_index(tree_node, constraints_indices[new_largest_ind])
                    .unwrap();
            }
        }
    }

    tree
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use portgraph::PortOffset;

    use crate::portgraph::indexing::PGIndexKey;

    use super::*;

    fn index_key(path_root: usize, path_length: usize) -> PGIndexKey {
        PGIndexKey::AlongPath {
            path_root,
            path_start_port: PortOffset::Outgoing(0),
            path_length,
        }
    }

    #[test]
    fn test_all_combinations() {
        let constraints = vec![
            (
                PGConstraint::try_new(
                    PGPredicate::<()>::IsNotEqual { n_other: 2 },
                    vec![index_key(0, 0), index_key(1, 1), index_key(2, 1)],
                )
                .unwrap(),
                0,
            ),
            (
                PGConstraint::try_new(
                    PGPredicate::IsNotEqual { n_other: 2 },
                    vec![index_key(0, 0), index_key(1, 1), index_key(1, 2)],
                )
                .unwrap(),
                1,
            ),
            (
                PGConstraint::try_new(
                    PGPredicate::IsNotEqual { n_other: 2 },
                    vec![index_key(0, 0), index_key(1, 1), index_key(1, 3)],
                )
                .unwrap(),
                2,
            ),
        ];
        let tree = all_combinations(&constraints);
        assert_eq!(tree.n_nodes(), 1 + 3 + 2 + 1 + 1);
        assert_debug_snapshot!(tree);
    }
}
