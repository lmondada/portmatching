use crate::{
    constraint::DetHeuristic,
    constraint_tree::{ConstraintTree, ToConstraintsTree},
    utils::sort_with_indices,
    Constraint,
};

use super::{predicate::CharacterPredicate, StringPatternPosition};

/// A constraint for matching a string using [StringPredicate]s.
pub type StringConstraint<K = StringPatternPosition> = Constraint<K, CharacterPredicate>;

impl<K: Copy + Ord> Ord for StringConstraint<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let mut self_bindings = self.required_bindings().to_vec();
        let mut other_bindings = other.required_bindings().to_vec();
        self_bindings.sort_by(|a, b| a.cmp(b).reverse());
        other_bindings.sort_by(|a, b| a.cmp(b).reverse());
        self_bindings.cmp(&other_bindings)
    }
}

impl<K: Copy + Ord> PartialOrd for StringConstraint<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Copy + Ord> ToConstraintsTree<K> for CharacterPredicate {
    fn to_constraints_tree(
        constraints: Vec<StringConstraint<K>>,
    ) -> ConstraintTree<StringConstraint<K>> {
        if constraints.is_empty() {
            return ConstraintTree::new();
        }
        let mut constraints = sort_with_indices(constraints);
        let first_constraint = &constraints[0].0;
        match *first_constraint.predicate() {
            CharacterPredicate::BindingEq => {
                // Only keep the first constraint
                constraints.truncate(1);
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
            }
        }
        ConstraintTree::with_children(constraints.into_iter().map(|(c, i)| (c, vec![i])))
    }
}

impl<K: Copy + Ord> DetHeuristic<K> for CharacterPredicate {
    fn make_det(constraints: &[&StringConstraint<K>]) -> bool {
        constraints.is_empty()
            || matches!(constraints[0].predicate(), CharacterPredicate::ConstVal(_))
    }
}
