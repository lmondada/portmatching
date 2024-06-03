use std::mem;

use itertools::Itertools;

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    constraint::Constraint,
    mutex_tree::ToMutuallyExclusiveTree,
    predicate::ArityPredicate,
    PatternID,
};

pub struct AutomatonBuilder<C> {
    /// A vector of patterns, made of a vector of constraints.
    patterns: Vec<Vec<C>>,
    /// The matcher being built
    matcher: ConstraintAutomaton<C>,
}

impl<C> AutomatonBuilder<C> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_constraints(patterns: Vec<Vec<C>>) -> Self {
        Self {
            patterns,
            matcher: ConstraintAutomaton::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: Vec<C>) {
        self.patterns.push(pattern);
    }
}

type Automaton<V, U, AP, FP> = ConstraintAutomaton<Constraint<V, U, AP, FP>>;

impl<V, U, AP, FP> AutomatonBuilder<Constraint<V, U, AP, FP>>
where
    AP: ArityPredicate,
    FP: ArityPredicate,
    Constraint<V, U, AP, FP>: Eq + Clone + ToMutuallyExclusiveTree,
{
    /// Construct the automaton.
    ///
    /// The returned automaton will be able to match `self.patterns` and will
    /// respect the automaton invariants: at every state and for any input, the transitions are mutually exclusive
    /// or an epsilon transition.
    ///
    /// The `make_det` predicate specifies the heuristic used to determine whether
    /// to turn a state into a deterministic one.
    pub fn build(
        mut self,
        make_det: impl for<'c> Fn(&[&'c Constraint<V, U, AP, FP>]) -> bool,
    ) -> (Automaton<V, U, AP, FP>, Vec<PatternID>) {
        // Construct a prefix tree by adding all constraints non-deterministically
        let pattern_ids = mem::take(&mut self.patterns)
            .into_iter()
            .map(|constraints| self.matcher.add_constraints(constraints))
            .collect_vec();

        // Traverse the prefix tree and do two things:
        // 1. Make the mutually exclusivity invariant hold:
        //    at every node for any input, at most one constraint may be satisfied
        // 2. Turn some of the states into deterministic transitions, according to
        //    `make_det`
        // Proceed from root to leaves.
        let mut curr_states = vec![self.matcher.root()];
        while let Some(state) = curr_states.pop() {
            // Make the mutually exclusivity invariant hold
            self.make_mutex(state);
            // Turn some of the states into deterministic transitions, according to
            // `make_det`
            let constraints = self.matcher.constraints(state).collect_vec();
            if make_det(&constraints) {
                self.matcher.make_det(state);
            }

            // Add all children to the stack
            curr_states.extend(self.matcher.children(state));
        }
        (self.matcher, pattern_ids)
    }

    /// Turn outgoing constraints at `state` into a mutually exclusive set.
    ///
    /// Use `ToMutuallyExclusiveTree` to turn the constraints into a mutually
    /// exclusive tree, which is inserted in place of `state`.
    ///
    /// This may insert epsilon transitions, i.e. edges with no associated
    /// constraint. These will always be last in the constraint order.
    fn make_mutex(&mut self, state: StateID) {
        assert!(!self.matcher.is_deterministic(state));

        if !self.matcher.constraints(state).any(|_| true) {
            // There are no constraints, already in a deterministic state
            return;
        }
        // Disconnect all children
        let constraints_children = self.matcher.drain_constraints(state).collect_vec();
        let old_fail_state = constraints_children
            .iter()
            .find_map(|(cons, child)| cons.is_none().then_some(*child));

        // Filter out None constraint
        let (constraints, children): (Vec<_>, Vec<_>) = constraints_children
            .into_iter()
            .filter_map(|(cons, child)| Some((cons?, child)))
            .unzip();

        // Organise constraints into a tree of mutually exclusive constraints
        let mutex_tree = Constraint::to_mutually_exclusive_tree(constraints.clone());
        assert!(mutex_tree.is_valid_tree());
        let mut added_constraints = vec![false; constraints.len()];

        // Traverse the mutex tree, making sure to
        //  - add new state to the matcher as we go
        //  - keep track of the matcher state corresponding to tree states
        //  - add edges to children when the constraint index is set
        let mut curr_states = vec![(mutex_tree.root(), state)];
        while let Some((tree_state, matcher_state)) = curr_states.pop() {
            for (child_tree_state, c) in mutex_tree.children(tree_state) {
                if let Some(index) = mutex_tree.constraint_index(child_tree_state) {
                    added_constraints[index] = true;
                    self.matcher.add_transition_known_child(
                        matcher_state,
                        children[index],
                        Some(constraints[index].clone()),
                    );
                } else {
                    let child_matcher_state = self
                        .matcher
                        .add_transition_unknown_child(matcher_state, Some(c.clone()));
                    curr_states.push((child_tree_state, child_matcher_state));
                }
            }
        }

        // All constraints that were not present in the mutex tree are added as
        // children of an epsilon transition at `state`.
        let not_added = (0..constraints.len())
            .filter(|&i| !added_constraints[i])
            .collect_vec();
        // Restore failed state (or create if necessary)
        if !not_added.is_empty() || old_fail_state.is_some() {
            let fail_state = if let Some(fail_state) = old_fail_state {
                self.matcher
                    .add_transition_known_child(state, fail_state, None);
                fail_state
            } else {
                self.matcher.add_transition_unknown_child(state, None)
            };
            // Add edges to children
            for i in not_added {
                self.matcher.add_transition_known_child(
                    fail_state,
                    children[i],
                    Some(constraints[i].clone()),
                );
            }
        }
    }
}

impl<C> Default for AutomatonBuilder<C> {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            matcher: ConstraintAutomaton::new(),
        }
    }
}

impl<C> FromIterator<Vec<C>> for AutomatonBuilder<C> {
    fn from_iter<T: IntoIterator<Item = Vec<C>>>(iter: T) -> Self {
        Self {
            patterns: iter.into_iter().collect(),
            matcher: ConstraintAutomaton::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constraint::tests::{assign_constraint, TestConstraint},
        mutex_tree::MutuallyExclusiveTree,
        ConstraintLiteral, HashSet,
    };

    use super::*;

    // Dummy ToMutuallyExclusiveTree implementation for TestConstraint
    impl ToMutuallyExclusiveTree for TestConstraint {
        fn to_mutually_exclusive_tree(preds: Vec<Self>) -> MutuallyExclusiveTree<Self> {
            // Only process the first two smallest constraints
            let (inds, preds): (Vec<_>, Vec<_>) = preds
                .into_iter()
                .enumerate()
                .sorted_by(|(_, p1), (_, p2)| p1.cmp(p2))
                .unzip();
            let first_two = preds.into_iter().take(2);
            let mut tree = MutuallyExclusiveTree::new();
            let new_children = tree.add_children(tree.root(), first_two).collect_vec();
            for (index, child) in inds.into_iter().zip(new_children) {
                tree.set_constraint_index(child, index);
            }
            tree
        }
    }

    #[test]
    fn test_build() {
        let a_constraint = assign_constraint("a", ConstraintLiteral::new_value(2));
        let p1 = vec![
            assign_constraint("x", ConstraintLiteral::new_value(1)),
            a_constraint.clone(),
        ];
        let b_constraint = assign_constraint("b", ConstraintLiteral::new_value(2));
        let p2 = vec![
            assign_constraint("x", ConstraintLiteral::new_value(1)),
            b_constraint.clone(),
        ];
        let c_constraint = assign_constraint("c", ConstraintLiteral::new_value(2));
        let p3 = vec![
            assign_constraint("x", ConstraintLiteral::new_value(1)),
            c_constraint.clone(),
        ];
        let d_constraint = assign_constraint("d", ConstraintLiteral::new_value(2));
        let p4 = vec![
            assign_constraint("x", ConstraintLiteral::new_value(1)),
            d_constraint.clone(),
        ];
        let builder = AutomatonBuilder::from_constraints(vec![p1, p2, p3, p4]);
        let (matcher, pattern_ids) = builder.build(|_| false);
        assert_eq!(matcher.graph.node_count(), 7);
        assert_eq!(
            pattern_ids,
            vec![PatternID(0), PatternID(1), PatternID(2), PatternID(3)]
        );
        let x_child = matcher.children(matcher.root()).exactly_one().ok().unwrap();

        // The two first patterns were kept at the root
        assert_eq!(
            matcher
                .transitions(x_child)
                .map(|t| { matcher.constraint(t) })
                .collect::<HashSet<_>>(),
            HashSet::from_iter([Some(&a_constraint), Some(&b_constraint), None])
        );
        // The remaining two patterns are children of an epsilon transition
        let epsilon = matcher.find_constraint(x_child, None).unwrap();
        let epsilon_child = matcher.next_state(epsilon);
        assert_eq!(
            matcher
                .transitions(epsilon_child)
                .map(|t| { matcher.constraint(t) })
                .collect::<HashSet<_>>(),
            HashSet::from_iter([Some(&c_constraint), Some(&d_constraint)])
        );
    }
}
