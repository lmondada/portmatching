use crate::{
    constraint::DetHeuristic,
    mutex_tree::{MutuallyExclusiveTree, ToConstraintsTree},
    utils::sort_with_indices,
    Constraint,
};

use super::{predicate::StringPredicate, StringIndexKey};

/// A constraint for matching a string using [StringPredicate]s.
pub type StringConstraint<K = StringIndexKey> = Constraint<K, StringPredicate>;

impl<K: Copy + Ord> StringConstraint<K> {
    /// The largest variable index in the constraint
    fn max_var(&self) -> K {
        *self.required_bindings().iter().max().unwrap()
    }
}

impl<K: Copy + Ord> Ord for StringConstraint<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.max_var().cmp(&other.max_var())
    }
}

impl<K: Copy + Ord> PartialOrd for StringConstraint<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: Copy + Ord> ToConstraintsTree for StringConstraint<K> {
    fn to_constraints_tree(constraints: Vec<Self>) -> MutuallyExclusiveTree<Self> {
        if constraints.is_empty() {
            return MutuallyExclusiveTree::new();
        }
        let mut constraints = sort_with_indices(constraints);
        let first_constraint = constraints[0].0.clone();
        match first_constraint.predicate() {
            StringPredicate::BindingEq => {
                // Only keep the first constraint
                constraints.truncate(1);
            }
            StringPredicate::ConstVal(_) => {
                // Checks of a variable against constant values are always mutually
                // exclusive
                constraints.retain(|(c, _)| {
                    matches!(c.predicate(), StringPredicate::ConstVal(_))
                        && c.required_bindings() == first_constraint.required_bindings()
                });
            }
        }
        MutuallyExclusiveTree::with_children(constraints.into_iter().map(|(c, i)| (c, vec![i])))
    }
}

impl<K: Copy + Ord> DetHeuristic for StringConstraint<K> {
    fn make_det(constraints: &[&Self]) -> bool {
        if constraints.is_empty() {
            return true;
        }
        matches!(constraints[0].predicate(), StringPredicate::ConstVal(_))
    }
}
