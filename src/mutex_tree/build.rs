//! Methods to simplify the construction of constraint trees.

use std::collections::VecDeque;

use itertools::Itertools;

use crate::Constraint;

use super::{ConditionedPredicate, MutuallyExclusiveTree};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct IndexedConstraint<C> {
    constraint: C,
    index: usize,
}

impl<C> From<(C, usize)> for IndexedConstraint<C> {
    fn from((constraint, index): (C, usize)) -> Self {
        Self { constraint, index }
    }
}

impl<C> From<IndexedConstraint<C>> for (C, usize) {
    fn from(IndexedConstraint { constraint, index }: IndexedConstraint<C>) -> Self {
        (constraint, index)
    }
}

struct QueueItem<'c, C> {
    next_constraint_index: usize,
    satisfied_constraints: Vec<&'c C>,
    tree_node: usize,
}

fn add_implied_constraints<'c, K, P: ConditionedPredicate<K>>(
    tree: &mut MutuallyExclusiveTree<Constraint<K, P>>,
    node: usize,
    satisfied_constraints: &mut Vec<&'c Constraint<K, P>>,
    next_constraint_index: &mut usize,
    constraints: &'c [(Constraint<K, P>, usize)],
) -> Option<Constraint<K, P>> {
    while *next_constraint_index < constraints.len() {
        let next_constraint = &constraints[*next_constraint_index].0;
        if let Some(conditioned_constraint) = P::conditioned(next_constraint, satisfied_constraints)
        {
            return Some(conditioned_constraint);
        }
        satisfied_constraints.push(next_constraint);
        tree.add_constraint_index(node, constraints[*next_constraint_index].1);
        *next_constraint_index += 1;
    }
    None
}

impl<C> MutuallyExclusiveTree<C> {
    /// Creates a constraint tree with a filtered subset of mutually exclusive
    /// `constraints`, under the assumption that the mutual exclusion relation
    /// is transitive.
    ///
    /// I.e. if A and B are mutually exclusive, and B and C are mutually exclusive,
    /// then A, B and C must be a mutually exclusive set.
    pub fn with_transitive_mutex(
        constraints: impl IntoIterator<Item = (C, usize)>,
        is_mutex: impl Fn(&C, &C) -> bool,
    ) -> Self {
        let mut constraints = constraints.into_iter();
        let Some((first, first_ind)) = constraints.next() else {
            return MutuallyExclusiveTree::new();
        };
        let constraints = constraints
            .filter(|(c, _)| is_mutex(&first, c))
            .map(|(c, i)| (c, vec![i]))
            .collect_vec();
        MutuallyExclusiveTree::with_children(
            [(first, vec![first_ind])].into_iter().chain(constraints),
        )
    }
}

impl<P: ConditionedPredicate<K>, K> MutuallyExclusiveTree<Constraint<K, P>> {
    /// Creates a constraint tree of all the conjunctions of subsets of the given
    /// constraints.
    ///
    /// Uses conditioned predicate to simplify constraints.
    pub fn with_powerset(constraints: Vec<(Constraint<K, P>, usize)>) -> Self {
        if constraints.is_empty() {
            return MutuallyExclusiveTree::new();
        }

        // Start by putting the constraints themselves. We will put combinations
        // of these constraints as descendants
        let mut tree = MutuallyExclusiveTree::new();

        // A queue of nodes in the tree to add children to
        let mut queue = VecDeque::from_iter([QueueItem {
            next_constraint_index: 0,
            satisfied_constraints: vec![],
            tree_node: tree.root(),
        }]);

        while let Some(item) = queue.pop_front() {
            let QueueItem {
                mut next_constraint_index,
                mut satisfied_constraints,
                mut tree_node,
            } = item;
            // Add all implied constraints and find the first constraint that
            // is not implied by the satisfied set
            let next_constraint = add_implied_constraints(
                &mut tree,
                tree_node,
                &mut satisfied_constraints,
                &mut next_constraint_index,
                &constraints,
            );
            let Some(next_constraint) = next_constraint else {
                // We've exhausted the list of constraints
                continue;
            };

            // Add two new items to the queue: one where the new constraint is
            // added to the satisfied set, and one where it is not.

            // The simple case: just ignore the new constraint and move on
            queue.push_back(QueueItem {
                next_constraint_index: next_constraint_index + 1,
                satisfied_constraints: satisfied_constraints.clone(),
                tree_node,
            });

            // The interesting case: add a child in tree with the new constraint
            tree_node = tree.add_child(tree_node, next_constraint);
            satisfied_constraints.push(&constraints[next_constraint_index].0);
            tree.add_constraint_index(tree_node, constraints[next_constraint_index].1);
            next_constraint_index += 1;

            queue.push_back(QueueItem {
                next_constraint_index,
                satisfied_constraints,
                tree_node,
            });
        }

        tree
    }
}
