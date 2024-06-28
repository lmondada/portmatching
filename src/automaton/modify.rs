use itertools::Itertools;

use super::{
    traced::TracedAutomaton, ConstraintAutomaton, State, StateID, Transition, TransitionID,
};
use crate::{utils::mark_last, PatternID};

/// Methods for modifying the automaton
///
/// Exposed as a trait so that the automaton builder can reuse the default
/// implementation but trace calls.
impl<C: Eq + Clone, I> ConstraintAutomaton<C, I> {
    /// Add a new disconnected non-deterministic state
    pub(super) fn add_non_det_node(&mut self) -> StateID {
        let node = self.graph.add_node(State {
            matches: vec![],
            deterministic: false,
            order: vec![],
        });
        StateID(node)
    }

    /// Set the deterministic flag of `state` to true and return whether it was
    /// already deterministic.
    pub(super) fn set_deterministic(&mut self, state: StateID) -> bool {
        let node = self.node_weight_mut(state);
        let was_det = node.deterministic;
        node.deterministic = true;
        was_det
    }

    /// Add a transition on the transition graph from `parent` to `child` with
    /// the given `constraint`.
    ///
    /// The edge is appended at the end of the edge list, with the caveat that
    /// the edge order invariant that `None` is always at the end must be maintained.
    /// Thus if a fail transition is present and `constraint` is not None, the
    /// new transition is added at the penultimate position.
    ///
    /// The `constraint` must not exist at `parent`.
    pub(super) fn append_edge(
        &mut self,
        StateID(parent): StateID,
        StateID(child): StateID,
        constraint: Option<C>,
    ) {
        if parent == child {
            // Transitions to itself are useless
            return;
        }

        // Transition constraints must be unique
        assert!(!self
            .graph
            .edges(parent)
            .any(|e| e.weight().constraint == constraint));

        let is_fail_transition = constraint.is_none();
        let new_edge = self
            .graph
            .add_edge(parent, child, Transition { constraint });
        self.append_order(TransitionID(new_edge), is_fail_transition);

        debug_assert!(self.check_edge_order_invariant(StateID(parent)));
    }

    /// Add a transition from `parent` to each child in `children` with the given
    /// `constraint`.
    ///
    /// The child of a constraint must be unique, so this requires a little trick:
    /// we add multiple children using a cascade of nodes with two children:
    /// a child with the `constraint` transition and a child with the `FAIL`
    /// transition, onto which all other children will be added.
    ///
    /// The `constraint` must not exist at `parent`.
    pub(super) fn append_edge_cascade(
        &mut self,
        mut parent: StateID,
        children: impl IntoIterator<Item = StateID>,
        constraint: Option<C>,
    ) {
        for (child, is_last) in mark_last(children) {
            self.append_edge(parent, child, constraint.clone());
            if !is_last {
                parent = self.add_transition(parent, None);
            }
        }
    }

    /// Add a transition from `parent` with the given `constraint`.
    ///
    /// If the transition already exists, this is a no-op and returns the existing
    /// child.
    /// Otherwise, a new child is created and a transition from `parent` to the
    /// new child is added.
    pub(super) fn add_transition(&mut self, parent: StateID, constraint: Option<C>) -> StateID {
        if let Some(transition) = self.find_constraint(parent, constraint.as_ref()) {
            // Transition exists, use it
            self.next_state(transition)
        } else {
            // Create a new state and new transition
            let child = self.add_non_det_node();
            self.append_edge(parent, child, constraint);
            child
        }
    }

    pub(super) fn add_match(&mut self, state: StateID, pattern: PatternID) {
        self.node_weight_mut(state).matches.push(pattern);
    }

    pub(crate) fn add_pattern(&mut self, constraints: impl IntoIterator<Item = C>, id: PatternID) {
        let mut curr_state = self.root();
        for constraint in constraints {
            curr_state = self.add_transition(curr_state, Some(constraint));
        }
        self.add_match(curr_state, id);
    }

    pub(super) fn drain_constraints(
        &mut self,
        state: StateID,
    ) -> impl Iterator<Item = (Option<C>, StateID)> + '_ {
        let transitions = self.transitions(state).collect_vec();
        transitions.into_iter().map(|transition| {
            let target = self.next_state(transition);
            let constraint = self.remove_transition(transition).constraint;
            (constraint, target)
        })
    }

    fn remove_transition(&mut self, transition: TransitionID) -> Transition<C> {
        self.remove_order(transition);
        self.graph
            .remove_edge(transition.0)
            .expect("invalid transition")
    }

    fn clone_state(&mut self, state: StateID) -> StateID {
        let new_weight = State {
            order: vec![],
            ..self.graph.node_weight(state.0).unwrap().clone()
        };
        let node = self.graph.add_node(new_weight);
        StateID(node)
    }

    /// Rewire incoming transitions to `state` through a new copy of `state`.
    ///
    /// Create a new state with the same outgoing transitions as `state` then
    /// remove the `incoming_transitions` from `state` and replace them with
    /// transitions to the new state.
    pub(super) fn split_state(
        &mut self,
        state: StateID,
        incoming_transitions: impl IntoIterator<Item = TransitionID>,
    ) -> StateID {
        let split_state = self.clone_state(state);

        // Copy outgoing transitions of `state` to `split_state`
        for transition in self.transitions(state).collect_vec() {
            let target = self.next_state(transition);
            self.append_edge(split_state, target, self.constraint(transition).cloned());
        }
        // Move transitions in `incoming_transitions` to `split_state`
        for transition in incoming_transitions {
            let source = self.parent(transition);
            debug_assert_eq!(self.next_state(transition), state);
            let Transition { constraint } = self.remove_transition(transition);
            self.append_edge(source, split_state, constraint);
        }
        split_state
    }

    pub(super) fn make_det(&mut self, state: StateID) {
        let mut traced = TracedAutomaton::new(state, self);
        traced.make_det(traced.initial_node());
        traced.zip();
        traced.apply_trace();
    }
}

// Small, private utils functions
impl<C: Eq + Clone, I> ConstraintAutomaton<C, I> {
    /// Adjust edge order after adding edge
    fn append_order(&mut self, new_transition: TransitionID, is_fail: bool) {
        let source = self.parent(new_transition);
        let order = &self.node_weight(source).order;
        let existing_fail = order.iter().position(|&e| self.constraint(e).is_none());
        // Promote order to &mut
        let order = &mut self.node_weight_mut(source).order;
        match (is_fail, existing_fail) {
            (_, None) => order.push(new_transition),
            (false, Some(ind)) => order.insert(ind, new_transition),
            (true, Some(_)) => unreachable!(),
        }
    }

    fn remove_order(&mut self, transition: TransitionID) {
        let source = self.parent(transition);
        let order = &mut self.node_weight_mut(source).order;
        let ind = order
            .iter()
            .position(|&e| e == transition)
            .expect("invalid transition");
        order.remove(ind);
    }

    fn node_weight_mut(&mut self, StateID(state): StateID) -> &mut State {
        self.graph.node_weight_mut(state).expect("invalid state")
    }
}

#[cfg(test)]
pub mod tests {
    use rstest::{fixture, rstest};

    use crate::{constraint::tests::TestConstraint, indexing::tests::TestIndexingScheme, HashSet};

    use super::*;

    /// An automaton with a X transition at the root and transitions
    /// [a,b,c,d] at the only child
    #[fixture]
    pub fn automaton() -> ConstraintAutomaton<TestConstraint, TestIndexingScheme> {
        let mut automaton = ConstraintAutomaton::new();
        let [constraint_root, constraint_a, constraint_b, constraint_c, constraint_d] =
            constraints();
        automaton.add_pattern([constraint_root.clone(), constraint_a], PatternID(0));
        automaton.add_pattern([constraint_root.clone(), constraint_b], PatternID(1));
        automaton.add_pattern([constraint_root.clone(), constraint_c], PatternID(2));
        automaton.add_pattern([constraint_root, constraint_d], PatternID(3));
        automaton
    }

    fn constraints() -> [TestConstraint; 5] {
        [
            TestConstraint::new(vec![0]),
            TestConstraint::new(vec![0, 4]),
            TestConstraint::new(vec![1, 4]),
            TestConstraint::new(vec![2, 4]),
            TestConstraint::new(vec![3, 4]),
        ]
    }

    fn root_child(automaton: &ConstraintAutomaton<TestConstraint, TestIndexingScheme>) -> StateID {
        let cs = automaton.children(automaton.root()).collect_vec();
        assert_eq!(cs.len(), 1);
        cs[0]
    }

    pub fn root_grandchildren() -> [StateID; 4] {
        [
            StateID(2.into()),
            StateID(3.into()),
            StateID(4.into()),
            StateID(5.into()),
        ]
    }

    #[rstest]
    fn test_add_constraints(automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        assert_eq!(automaton.graph.node_count(), 6);
        assert_eq!(automaton.transitions(automaton.root()).count(), 1);
        assert_eq!(automaton.transitions(root_child(&automaton)).count(), 4);
    }

    #[rstest]
    fn test_drain_constraints(
        mut automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>,
    ) {
        let [a_child, b_child, c_child, d_child] = root_grandchildren();
        let [_, constraint_a, constraint_b, constraint_c, constraint_d] = constraints();
        let drained: HashSet<_> = automaton
            .drain_constraints(root_child(&automaton))
            .collect();
        assert_eq!(
            drained,
            HashSet::from_iter([
                (Some(constraint_a), a_child),
                (Some(constraint_b), b_child),
                (Some(constraint_c), c_child),
                (Some(constraint_d), d_child)
            ])
        );
    }

    #[rstest]
    fn test_make_det_noop(mut automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        let automaton2 = automaton.clone();
        automaton.make_det(root_child(&automaton));
        assert_eq!(automaton.graph.node_count(), automaton2.graph.node_count());
        assert_eq!(automaton.graph.edge_count(), automaton2.graph.edge_count());
    }

    #[rstest]
    fn test_make_det(mut automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        let x_child = root_child(&automaton);
        let [a_child, b_child, c_child, d_child] = root_grandchildren();

        // Add a FAIL transition from x_child to a new state
        let fail_child = automaton.add_transition(x_child, None);
        // Add a common constraint to the fail child and a_child
        let common_constraint = TestConstraint::new(vec![7, 8]);
        let post_fail = automaton.add_transition(fail_child, Some(common_constraint.clone()));
        let post_a = automaton.add_transition(a_child, Some(common_constraint.clone()));
        // Add a second common constraint to post_fail
        let common_constraint2 = TestConstraint::new(vec![77, 8]);
        let post_post_fail = automaton.add_transition(post_fail, Some(common_constraint2.clone()));

        automaton.make_det(x_child);

        // Now `common_constraint` should be on all children, pointing to
        // `post_fail` for b_child, c_child and d_child. For a_child, the `post_a`
        // should have a `common_constraint2` transition to `post_post_fail`
        for child in [b_child, c_child, d_child] {
            assert_eq!(automaton.transitions(child).count(), 1);
            let transition = automaton.transitions(child).next().unwrap();
            assert_eq!(automaton.constraint(transition), Some(&common_constraint));
            assert_eq!(automaton.next_state(transition), post_fail);
        }
        let child = a_child;
        let transition = automaton.transitions(child).next().unwrap();
        assert_eq!(automaton.constraint(transition), Some(&common_constraint));
        assert_eq!(automaton.next_state(transition), post_a);

        let grandchild = post_a;
        let transition = automaton.transitions(grandchild).next().unwrap();
        assert_eq!(automaton.constraint(transition), Some(&common_constraint2));
        assert_eq!(automaton.next_state(transition), post_post_fail);
    }
}
