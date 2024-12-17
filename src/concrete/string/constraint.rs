use crate::{
    constraint_tree::{ConstraintTree, ToConstraintsTree},
    utils::sort_with_indices,
    Constraint,
};

use super::{predicate::CharacterPredicate, StringPatternPosition};

/// A constraint for matching a string using [StringPredicate]s.
pub type StringConstraint<K = StringPatternPosition> = Constraint<K, CharacterPredicate>;

impl<K: Copy + Ord> ToConstraintsTree<K> for CharacterPredicate {
    fn to_constraints_tree(
        constraints: Vec<StringConstraint<K>>,
    ) -> ConstraintTree<StringConstraint<K>> {
        if constraints.is_empty() {
            return ConstraintTree::with_make_det(true);
        }
        let mut constraints = sort_with_indices(constraints);
        let first_constraint = &constraints[0].0;
        let make_det = match *first_constraint.predicate() {
            CharacterPredicate::BindingEq => {
                // Only keep the first constraint
                constraints.truncate(1);
                false
            }
            CharacterPredicate::ConstVal(_) => {
                // Checks of a variable against constant values are always mutually
                // exclusive
                let &[first_cst_binding] = first_constraint.required_bindings() else {
                    panic!()
                };
                constraints.retain(|(c, _)| {
                    matches!(c.predicate(), CharacterPredicate::ConstVal(_))
                        && c.required_bindings() == [first_cst_binding]
                });
                true
            }
        };
        let mut tree =
            ConstraintTree::with_children(constraints.into_iter().map(|(c, i)| (c, vec![i])));
        tree.set_make_det(make_det);
        tree
    }
}
