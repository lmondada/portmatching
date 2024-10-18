//! Logic to compute mutually exclusive PGConstraints.

use crate::{constraint_tree::ConstraintTree, portgraph::predicate::PGPredicate};

use super::PGConstraint;

fn fst_required_binding_eq(a: &PGConstraint, b: &PGConstraint) -> bool {
    a.required_bindings()[0] == b.required_bindings()[0]
}
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
) -> ConstraintTree<PGConstraint> {
    // Assume constraints is not empty
    let first_constraint = constraints[0].0.clone();

    match first_constraint.predicate() {
        PGPredicate::IsNotEqual { .. } => {
            let constraints = constraints.into_iter().filter(|(c, _)| {
                // We can only turn IsNotEqual constraints into mutex predicates
                // if they act on the same variable
                matches!(c.predicate(), PGPredicate::IsNotEqual { .. })
                    && fst_required_binding_eq(c, &first_constraint)
            });
            ConstraintTree::with_powerset(constraints.collect())
        }
        PGPredicate::HasNodeWeight(..) | PGPredicate::IsConnected { .. } => {
            ConstraintTree::with_transitive_mutex(constraints, |a, b| {
                match (a.predicate(), b.predicate()) {
                    (PGPredicate::HasNodeWeight(..), PGPredicate::HasNodeWeight(..)) => {
                        fst_required_binding_eq(a, b)
                    }
                    (
                        PGPredicate::IsConnected {
                            left_port: lp_a, ..
                        },
                        PGPredicate::IsConnected {
                            left_port: lp_b, ..
                        },
                    ) => lp_a == lp_b && fst_required_binding_eq(a, b),
                    _ => false,
                }
            })
        }
    }
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
        let tree = ConstraintTree::with_powerset(constraints);
        assert_eq!(tree.n_nodes(), 1 + 3 + 2 + 1 + 1);
        assert_debug_snapshot!(tree);
    }
}
