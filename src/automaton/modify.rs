use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph};

use crate::PatternID;

use super::{ConstraintAutomaton, State, StateID, Transition};

impl<C: Eq + Clone> ConstraintAutomaton<C> {
    /// Add a transition from `parent` to `child` with the given `constraint`.
    ///
    /// Note that it could be that such a transition already exists at `parent`
    /// and points to `actual_child` instead of `child`. In that case,
    /// all transitions at `child` are added to `actual_child`, so that in effect
    /// the automaton "behaves" as if there is a transition from `parent` to
    /// `child`.
    ///
    /// In such a case, the transition adding will cascade recursively, so in
    /// some cases this might be expensive.
    pub(super) fn add_transition_known_child(
        &mut self,
        parent: StateID,
        child: StateID,
        constraint: Option<C>,
    ) {
        match self.find_constraint(parent, constraint.as_ref()) {
            Some(transition) => {
                // Transition already exists, we must recurse
                let actual_child = self.next_state(transition);
                self.copy_constraints(child, actual_child);
            }
            None => {
                // Add edge.
                let transition = Transition { constraint };
                self.graph.add_edge(parent.0, child.0, transition);
            }
        }
    }

    /// Add a transition from `parent` with the given `constraint`.
    ///
    /// If the transition already exists, this returns the existing child.
    /// Otherwise, a new child is created and a transition from `parent` to the
    /// new child is added.
    pub(super) fn add_transition_unknown_child(
        &mut self,
        parent: StateID,
        constraint: Option<C>,
    ) -> StateID {
        if let Some(transition) = self.find_constraint(parent, constraint.as_ref()) {
            // Transition exists, use it
            self.next_state(transition)
        } else {
            // Create a new state and new transition
            let child = self.add_non_det_node();
            self.add_transition_known_child(parent, child, constraint);
            child
        }
    }

    pub(super) fn add_match(&mut self, StateID(state): StateID, pattern: PatternID) {
        graph_node_weight_mut(&mut self.graph, state)
            .matches
            .push(pattern);
    }

    pub(crate) fn add_constraints(
        &mut self,
        constraints: impl IntoIterator<Item = C>,
    ) -> PatternID {
        let mut curr_state = self.root();
        for constraint in constraints {
            curr_state = self.add_transition_unknown_child(curr_state, Some(constraint));
        }
        let id = PatternID(self.n_patterns);
        self.add_match(curr_state, id);
        self.n_patterns += 1;
        id
    }

    pub(super) fn drain_constraints(
        &mut self,
        state: StateID,
    ) -> impl Iterator<Item = (Option<C>, StateID)> + '_ {
        let transitions = self.transitions(state).collect_vec();
        transitions.into_iter().map(|transition| {
            let target = self.next_state(transition);
            let constraint = self
                .graph
                .remove_edge(transition.0)
                .expect("invalid transition")
                .constraint;
            (constraint, target)
        })
    }

    /// Turn `state` into a deterministic state
    ///
    /// This assumes that all transitions from `state` are mutually exclusive,
    /// with the exception of the epsilon transition. The state is made
    /// deterministic by adding all transitions at the child of the epsilon, if
    /// it exists, to every other child of `state`.
    ///
    /// This will turn the epsilon transition into a FAIL (fallback) transition.
    pub(super) fn make_det(&mut self, state: StateID) {
        // Change the flag
        let det_flag = &mut graph_node_weight_mut(&mut self.graph, state.0).deterministic;
        if *det_flag {
            // Already deterministic
            return;
        } else {
            *det_flag = true;
        }

        // Epsilon transition becomes FAIL (fallback) transition
        let Some(fail_edge) = self.find_constraint(state, None) else {
            // There is no epsilon transition => `state` is already deterministic
            return;
        };
        let fail_state = self.next_state(fail_edge);

        // Add all constraint transitions of the FAIL state to every other child
        let non_fail_children = self
            .children(state)
            .filter(|&c| c != fail_state)
            .collect_vec();
        for child in non_fail_children {
            self.copy_constraints(fail_state, child);
        }
    }

    /// Copy all constraints on state `from` to state `to`.
    ///
    /// If a constraint of `from` already exists in `to`, then recursively copy
    /// the children of the constraint.
    fn copy_constraints(&mut self, from: StateID, to: StateID) {
        let transitions = self.transitions(from).collect_vec();
        for transition in transitions {
            let from_child = self.next_state(transition);
            let constraint = self.constraint(transition).cloned();
            self.add_transition_known_child(to, from_child, constraint);
        }
    }

    fn add_non_det_node(&mut self) -> StateID {
        let node = self.graph.add_node(State {
            matches: vec![],
            deterministic: false,
        });
        StateID(node)
    }
}

fn graph_node_weight_mut<N, E>(graph: &mut StableGraph<N, E>, state: NodeIndex) -> &mut N {
    graph.node_weight_mut(state).expect("invalid state")
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use crate::{
        constraint::tests::{assign_constraint, filter_constraint, TestConstraint},
        ConstraintLiteral, HashSet,
    };

    use super::*;

    /// An automaton with a X transition at the root and transitions
    /// [a,b,c,d] at the only child
    #[fixture]
    fn automaton() -> ConstraintAutomaton<TestConstraint> {
        let mut automaton = ConstraintAutomaton::new();
        let [constraint_root, constraint_a, constraint_b, constraint_c, constraint_d] =
            constraints();
        automaton.add_constraints([constraint_root.clone(), constraint_a]);
        automaton.add_constraints([constraint_root.clone(), constraint_b]);
        automaton.add_constraints([constraint_root.clone(), constraint_c]);
        automaton.add_constraints([constraint_root, constraint_d]);
        automaton
    }

    fn constraints() -> [TestConstraint; 5] {
        [
            assign_constraint("x", ConstraintLiteral::new_value(2)),
            assign_constraint("a", ConstraintLiteral::new_variable("x".to_string())),
            assign_constraint("b", ConstraintLiteral::new_variable("x".to_string())),
            assign_constraint("c", ConstraintLiteral::new_variable("x".to_string())),
            assign_constraint("d", ConstraintLiteral::new_variable("x".to_string())),
        ]
    }

    fn root_child(automaton: &ConstraintAutomaton<TestConstraint>) -> StateID {
        let cs = automaton.children(automaton.root()).collect_vec();
        assert_eq!(cs.len(), 1);
        cs[0]
    }

    fn root_grandchildren() -> [StateID; 4] {
        [
            StateID(2.into()),
            StateID(3.into()),
            StateID(4.into()),
            StateID(5.into()),
        ]
    }

    #[rstest]
    fn test_add_constraints(automaton: ConstraintAutomaton<TestConstraint>) {
        assert_eq!(automaton.graph.node_count(), 6);
        assert_eq!(automaton.transitions(automaton.root()).count(), 1);
        assert_eq!(automaton.transitions(root_child(&automaton)).count(), 4);
    }

    #[rstest]
    fn test_drain_constraints(mut automaton: ConstraintAutomaton<TestConstraint>) {
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
    fn test_add_known_child(mut automaton: ConstraintAutomaton<TestConstraint>) {
        let [a_child, _, c_child, d_child] = root_grandchildren();
        // Add a constraint to C
        let new_c = filter_constraint(
            ConstraintLiteral::new_variable("x".to_string()),
            ConstraintLiteral::new_value(2),
        );
        automaton.add_transition_unknown_child(c_child, Some(new_c.clone()));

        // Add a `None` constraint A -> D
        automaton.add_transition_known_child(a_child, d_child, None);
        assert!(automaton.find_constraint(a_child, None).is_some());

        // Try adding a `None` constraint A -> C, will instead recurse and add
        // C's constraints to D
        assert_eq!(automaton.transitions(d_child).count(), 0);
        automaton.add_transition_known_child(a_child, c_child, None);
        let d_transition = automaton.transitions(d_child).next().unwrap();
        assert_eq!(automaton.constraint(d_transition), Some(&new_c));
    }

    #[rstest]
    fn test_make_det_noop(mut automaton: ConstraintAutomaton<TestConstraint>) {
        let automaton2 = automaton.clone();
        automaton.make_det(root_child(&automaton));
        assert_eq!(automaton.graph.node_count(), automaton2.graph.node_count());
        assert_eq!(automaton.graph.edge_count(), automaton2.graph.edge_count());
    }

    #[rstest]
    fn test_make_det(mut automaton: ConstraintAutomaton<TestConstraint>) {
        let x_child = root_child(&automaton);
        let [a_child, b_child, c_child, d_child] = root_grandchildren();

        // Add a FAIL transition from x_child to a new state
        let fail_child = automaton.add_transition_unknown_child(x_child, None);
        // Add a common constraint to the fail child and a_child
        let common_constraint = filter_constraint(
            ConstraintLiteral::new_variable("common".to_string()),
            ConstraintLiteral::new_value(2),
        );
        let post_fail =
            automaton.add_transition_unknown_child(fail_child, Some(common_constraint.clone()));
        let post_a =
            automaton.add_transition_unknown_child(a_child, Some(common_constraint.clone()));
        // Add a second common constraint to post_fail
        let common_constraint2 = filter_constraint(
            ConstraintLiteral::new_variable("common2".to_string()),
            ConstraintLiteral::new_value(2),
        );
        let post_post_fail =
            automaton.add_transition_unknown_child(post_fail, Some(common_constraint2.clone()));

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
