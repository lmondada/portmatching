use std::mem;

use itertools::Itertools;
use petgraph::graph::NodeIndex;

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    constraint::Constraint,
    mutex_tree::ToConstraintsTree,
    PatternID,
};

use super::TransitionID;

/// Create constraint automata from lists of patterns, given by lists of
/// constraints.
pub struct AutomatonBuilder<C, I> {
    /// A vector of patterns, each expressed as a vector of constraints.
    patterns: Vec<Vec<C>>,
    /// The matcher being built
    matcher: ConstraintAutomaton<C, I>,
}

impl<C: Eq + Clone, I> AutomatonBuilder<C, I> {
    /// Construct an empty automaton builder.
    pub fn new() -> Self
    where
        I: Default,
    {
        Self::default()
    }

    /// Construct an automaton builder from a list of patterns, given by lists of
    /// constraints.
    ///
    /// Use `I::default()` as the indexing scheme.
    pub fn from_constraints(patterns: Vec<Vec<C>>) -> Self
    where
        I: Default,
    {
        Self {
            patterns,
            matcher: ConstraintAutomaton::new(),
        }
    }

    /// Construct an automaton builder from a list of patterns with a custom
    /// indexing scheme.
    pub fn from_constraints_with_index_scheme(patterns: Vec<Vec<C>>, host_indexing: I) -> Self {
        Self {
            patterns,
            matcher: ConstraintAutomaton::with_indexing_scheme(host_indexing),
        }
    }

    /// Add a pattern to the automaton builder.
    pub fn add_pattern(&mut self, pattern: Vec<C>) {
        self.patterns.push(pattern);
    }
}

type Automaton<K, P, I> = ConstraintAutomaton<Constraint<K, P>, I>;

impl<K, P, I> AutomatonBuilder<Constraint<K, P>, I>
where
    Constraint<K, P>: Eq + Clone + ToConstraintsTree,
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
        mut make_det: impl for<'c> FnMut(&[&'c Constraint<K, P>]) -> bool,
    ) -> (Automaton<K, P, I>, Vec<PatternID>) {
        // Construct a prefix tree by adding all constraints non-deterministically
        let pattern_ids = mem::take(&mut self.patterns)
            .into_iter()
            .enumerate()
            .map(|(id, constraints)| {
                let id = PatternID(id);
                self.matcher.add_pattern(constraints, id);
                id
            })
            .collect_vec();

        // Traverse the prefix tree from root to leaves and do two things:
        // 1. Make the mutually exclusivity invariant hold:
        //    at every node for any input, at most one constraint may be satisfied
        // 2. Turn some of the states into deterministic transitions, according to
        //    `make_det`

        // With this `toposort` we are allowed to add vertices and edges
        let mut traverser = self.matcher.toposort();
        while let Some(state) = traverser.next(&self.matcher) {
            // Make the mutually exclusivity invariant hold
            self.make_mutex(state);
            // Turn some of the states into deterministic transitions, according to
            // `make_det`
            let constraints = self.matcher.constraints(state).collect_vec();
            if make_det(&constraints) {
                self.matcher.make_det(state);
            }
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
        let mutex_tree = Constraint::to_constraints_tree(constraints.clone());
        let mut added_constraints = vec![false; constraints.len()];

        // Traverse the mutex tree, making sure to
        //  - add new state to the matcher as we go
        //  - keep track of the matcher state corresponding to tree states
        //  - add edges to children when the constraint index is set
        let mut curr_states = vec![(mutex_tree.root(), state)];
        while let Some((tree_state, matcher_state)) = curr_states.pop() {
            for (child_tree_state, c) in mutex_tree.children(tree_state) {
                let indices = mutex_tree.constraint_indices(child_tree_state);
                for &index in indices {
                    added_constraints[index] = true;
                }
                // It is always safe to add transitions in `make_mutex` without
                // tracing, since we only add edges to states that we just created
                // or drained all transitions from
                if !indices.is_empty() {
                    self.matcher.append_edge_cascade(
                        matcher_state,
                        indices.iter().map(|&ind| children[ind]),
                        Some(c.clone()),
                    );
                } else {
                    let child_matcher_state =
                        self.matcher.add_transition(matcher_state, Some(c.clone()));
                    curr_states.push((child_tree_state, child_matcher_state));
                }
            }
        }

        // All constraints that were not present in the mutex tree are added as
        // children of an epsilon transition at `state`.
        let not_added = (0..constraints.len())
            .filter(|&i| !added_constraints[i])
            .collect_vec();

        // Add/hide any constraints that were not added under a fail transition
        if !not_added.is_empty() {
            let fail_state = self.matcher.create_fail_state(state);
            // Add edges to children
            for i in not_added {
                self.matcher
                    .append_edge(fail_state, children[i], Some(constraints[i].clone()));
            }
        }

        // And finally, restore original fail state
        if let Some(old_fail_state) = old_fail_state {
            self.matcher.restore_fail_state(state, old_fail_state);
        }
    }
}

// Small, private utils functions
impl<C: Eq + Clone, I> ConstraintAutomaton<C, I> {
    /// Add a new fail state.
    ///
    /// Recursively traverse existing fail transitions until a new one can
    /// be created.
    fn create_fail_state(&mut self, state: StateID) -> StateID {
        let last_fail_state = self.find_last_fail_state(state);
        self.add_transition(last_fail_state, None)
    }

    /// Restore the original fail state.
    ///
    /// The same as `create_fail_state`, but linking to `old_fail_state` instead
    /// of creating a new state.
    fn restore_fail_state(&mut self, state: StateID, old_fail_state: StateID) {
        let last_fail_state = self.find_last_fail_state(state);
        self.append_edge(last_fail_state, old_fail_state, None);
    }

    fn find_last_fail_state(&mut self, mut state: StateID) -> StateID {
        loop {
            if let Some(transition) = self.find_fail_transition(state) {
                // Recurse down the Fail-transition chain
                state = self.next_state(transition);
            } else {
                // Found a `state` with no Fail transition!
                break state;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracedStateID {
    state: StateID,
    node: NodeIndex,
}

impl From<TracedStateID> for StateID {
    fn from(state: TracedStateID) -> Self {
        state.state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracedTransitionID {
    transition: TransitionID,
    parent: NodeIndex,
}

impl From<TracedTransitionID> for TransitionID {
    fn from(transition: TracedTransitionID) -> Self {
        transition.transition
    }
}

impl<C, I: Default> Default for AutomatonBuilder<C, I> {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            matcher: ConstraintAutomaton::new(),
        }
    }
}

impl<C, I: Default> FromIterator<Vec<C>> for AutomatonBuilder<C, I> {
    fn from_iter<T: IntoIterator<Item = Vec<C>>>(iter: T) -> Self {
        Self {
            patterns: iter.into_iter().collect(),
            matcher: ConstraintAutomaton::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint::tests::TestConstraint, mutex_tree::MutuallyExclusiveTree};

    use super::*;

    // Dummy ToMutuallyExclusiveTree implementation for TestConstraint
    impl ToConstraintsTree for TestConstraint {
        fn to_constraints_tree(preds: Vec<Self>) -> MutuallyExclusiveTree<Self> {
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
                tree.add_constraint_index(child, index).unwrap();
            }
            tree
        }
    }

    // #[test]
    // fn test_build() {
    //     let a_constraint = assign_constraint("a", ConstraintLiteral::new_value(2));
    //     let p1 = vec![
    //         assign_constraint("x", ConstraintLiteral::new_value(1)),
    //         a_constraint.clone(),
    //     ];
    //     let b_constraint = assign_constraint("b", ConstraintLiteral::new_value(2));
    //     let p2 = vec![
    //         assign_constraint("x", ConstraintLiteral::new_value(1)),
    //         b_constraint.clone(),
    //     ];
    //     let c_constraint = assign_constraint("c", ConstraintLiteral::new_value(2));
    //     let p3 = vec![
    //         assign_constraint("x", ConstraintLiteral::new_value(1)),
    //         c_constraint.clone(),
    //     ];
    //     let d_constraint = assign_constraint("d", ConstraintLiteral::new_value(2));
    //     let p4 = vec![
    //         assign_constraint("x", ConstraintLiteral::new_value(1)),
    //         d_constraint.clone(),
    //     ];
    //     let builder = AutomatonBuilder::from_constraints(vec![p1, p2, p3, p4]);
    //     let (matcher, pattern_ids) = builder.build(|_| false);
    //     assert_eq!(matcher.graph.node_count(), 7);
    //     assert_eq!(
    //         pattern_ids,
    //         vec![PatternID(0), PatternID(1), PatternID(2), PatternID(3)]
    //     );
    //     let x_child = matcher.children(matcher.root()).exactly_one().ok().unwrap();

    //     // The two first patterns were kept at the root
    //     assert_eq!(
    //         matcher
    //             .transitions(x_child)
    //             .map(|t| { matcher.constraint(t) })
    //             .collect::<HashSet<_>>(),
    //         HashSet::from_iter([Some(&a_constraint), Some(&b_constraint), None])
    //     );
    //     // The remaining two patterns are children of an epsilon transition
    //     let epsilon = matcher.find_constraint(x_child, None).unwrap();
    //     let epsilon_child = matcher.next_state(epsilon);
    //     assert_eq!(
    //         matcher
    //             .transitions(epsilon_child)
    //             .map(|t| { matcher.constraint(t) })
    //             .collect::<HashSet<_>>(),
    //         HashSet::from_iter([Some(&c_constraint), Some(&d_constraint)])
    //     );
    // }
}
