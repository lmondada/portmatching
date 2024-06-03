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

    pub(super) fn add_non_det_node(&mut self) -> StateID {
        let node = self.graph.add_node(State {
            matches: vec![],
            deterministic: false,
        });
        StateID(node)
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
}

fn graph_node_weight_mut<N, E>(graph: &mut StableGraph<N, E>, state: NodeIndex) -> &mut N {
    graph.node_weight_mut(state).expect("invalid state")
}
