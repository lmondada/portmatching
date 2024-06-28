//! Trace edits when building the automaton.
//!
//! This is useful to find when states must be split to avoid the following
//! situation: suppose we can reach state C from both A and B. If we now make
//! D reachable from A by adding a transition from C to D, then D is also
//! reachable from B, which may not be desirable! In that case, state C must
//! be split into two states, one reachable from A, and one from B.

use std::collections::BTreeSet;

use delegate::delegate;
use itertools::Itertools;

use super::{ConstraintAutomaton, StateID, TransitionID};
use crate::{
    utils::{TracedNode, Tracer},
    PatternID,
};

pub struct TracedAutomaton<'m, C, I> {
    /// The automaton being traced
    automaton: &'m mut ConstraintAutomaton<C, I>,
    /// The tracer used to trace the automaton
    tracer: Tracer<StateID, TransitionID, TraceAction<C>>,
}

struct TracedTransition {
    /// The transition ID
    transition: TransitionID,
    /// The state ID
    parent: TracedNode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TraceAction<C> {
    AddEdge { to: StateID, constraint: Option<C> },
    AddMatches { matches: Vec<PatternID> },
}

impl<'m, C: Eq + Clone, I> TracedAutomaton<'m, C, I> {
    /// Create a new traced automaton
    pub fn new(initial_node: StateID, automaton: &'m mut ConstraintAutomaton<C, I>) -> Self {
        let tracer = Tracer::new(initial_node);
        Self { automaton, tracer }
    }

    /// Apply the traced changes to the matcher.
    ///
    /// This will add the edges that have been traced; if adding the edge
    /// to the current node in the graph would result in "crosstalk", i.e.
    /// new transitions that should not be allowed, then nodes are split.
    pub fn apply_trace(self) {
        let Self { automaton, tracer } = self;

        // First, duplicate states in `matcher` until there is an injective
        // map of the trace into matcher
        for traced_node in tracer.toposort() {
            let Some(state) = tracer.node(traced_node) else {
                continue;
            };
            let traced_ids: BTreeSet<_> = tracer.traced_incoming(traced_node).collect();
            let (traced, non_traced): (Vec<_>, Vec<_>) = automaton
                .incoming_transitions(state)
                .partition(|e| traced_ids.contains(e));
            if !traced.is_empty() && !non_traced.is_empty() {
                automaton.split_state(state, non_traced);
            }
        }

        // Now add the edges to the matcher (must be the incoming edges of terminal)
        for (from, action) in tracer.into_actions() {
            match action {
                TraceAction::AddEdge { to, constraint } => {
                    automaton.append_edge(from, to, constraint);
                }
                TraceAction::AddMatches { matches } => {
                    for m in matches {
                        automaton.add_match(from, m);
                    }
                }
            }
        }
    }

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
    fn add_transition_known_child(
        &mut self,
        parent: TracedNode,
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
                self.tracer.add_action(
                    TraceAction::AddEdge {
                        to: child,
                        constraint,
                    },
                    parent,
                );
            }
        }
    }

    delegate! {
        to self.tracer {
            pub fn zip(&mut self);
            pub fn initial_node(&self) -> TracedNode;
        }
    }

    /// Copy all constraints on state `from` to state `to`.
    ///
    /// If a constraint of `from` already exists in `to`, then recursively copy
    /// the children of the constraint.
    ///
    /// Also adds all matches of `from` to `to`.
    fn copy_constraints(&mut self, from: StateID, to: TracedNode) {
        let transitions = self.automaton.transitions(from).collect_vec();
        let matches = self
            .automaton
            .matches(from)
            .iter()
            .copied()
            .filter(|m| {
                !self
                    .automaton
                    .matches(self.tracer.node(to).unwrap())
                    .contains(m)
            })
            .collect();
        self.tracer
            .add_action(TraceAction::AddMatches { matches }, to);
        for transition in transitions {
            let from_child = self.automaton.next_state(transition);
            let constraint = self.automaton.constraint(transition).cloned();
            self.add_transition_known_child(to, from_child, constraint);
        }
    }

    /// Turn `state` into a deterministic state
    ///
    /// This assumes that all transitions from `state` are mutually exclusive,
    /// with the exception of the epsilon transition. The state is made
    /// deterministic by adding all transitions at the child of the epsilon, if
    /// it exists, to every other child of `state`.
    ///
    /// This will turn the epsilon transition into a FAIL (fallback) transition.
    pub(super) fn make_det(&mut self, state: TracedNode) {
        let state_id = self.tracer.node(state).expect("Unexpected terminal node");

        // Set the deterministic flag, return if already set
        if self.automaton.set_deterministic(state_id) {
            // Already deterministic
            return;
        }

        // Epsilon transition becomes FAIL (fallback) transition
        let Some(fail_edge) = self.automaton.find_constraint(state_id, None) else {
            // There is no epsilon transition => `state` is already deterministic
            return;
        };
        let fail_state = self.automaton.next_state(fail_edge);

        // Add all constraint transitions of the FAIL state to every other child
        let non_fail_children = self
            .children(state)
            .into_iter()
            .filter(|&c| self.tracer.node(c).unwrap() != fail_state)
            .collect_vec();
        for child in non_fail_children {
            self.copy_constraints(fail_state, child);
        }
    }

    fn next_state(&mut self, transition: TracedTransition) -> TracedNode {
        let TracedTransition { transition, parent } = transition;
        let to_state = self.automaton.next_state(transition);
        self.tracer.traverse_edge(transition, parent, to_state)
    }

    fn find_constraint(
        &self,
        state: TracedNode,
        constraint: Option<&C>,
    ) -> Option<TracedTransition> {
        let state_id = self
            .tracer
            .node(state)
            .expect("Cannot find_constraint for terminal node");
        let transition = self.automaton.find_constraint(state_id, constraint)?;
        Some(TracedTransition {
            transition,
            parent: state,
        })
    }

    fn children(&mut self, state: TracedNode) -> Vec<TracedNode> {
        let state_id = self.tracer.node(state).expect("Unexpected terminal node");
        let transitions = self.automaton.transitions(state_id).collect_vec();
        transitions
            .into_iter()
            .map(|transition| {
                self.next_state(TracedTransition {
                    transition,
                    parent: state,
                })
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::automaton::modify::tests::{automaton, root_grandchildren};
    use crate::constraint::tests::TestConstraint;
    use crate::indexing::tests::TestIndexingScheme;

    use super::*;

    #[rstest]
    fn test_add_known_child(
        mut automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>,
    ) {
        let [a_child, _, c_child, d_child] = root_grandchildren();
        // Add a constraint to C
        let new_c = TestConstraint::new(vec![3]);
        automaton.add_transition(c_child, Some(new_c.clone()));

        // Add a `None` constraint A -> D
        let mut traced = TracedAutomaton::new(a_child, &mut automaton);
        traced.add_transition_known_child(traced.initial_node(), d_child, None);
        traced.apply_trace();
        // A transition was added to `a_child` that leads to `d_child`
        assert!(automaton.find_constraint(a_child, None).is_some());
        assert_eq!(automaton.incoming_transitions(d_child).count(), 2);

        // Try adding a `None` constraint A -> C, will instead recurse, split D
        // into two and add C's constraints to the copy of D
        assert_eq!(automaton.transitions(d_child).count(), 0);
        let mut traced = TracedAutomaton::new(a_child, &mut automaton);
        traced.add_transition_known_child(traced.initial_node(), c_child, None);
        traced.apply_trace();
        assert_eq!(automaton.incoming_transitions(d_child).count(), 1);
        let d_child_copy = {
            let new_transition = automaton.find_constraint(a_child, None).unwrap();
            automaton.next_state(new_transition)
        };
        assert_eq!(automaton.incoming_transitions(d_child_copy).count(), 1);
        let d_transition = automaton.transitions(d_child_copy).next().unwrap();
        assert_eq!(automaton.constraint(d_transition), Some(&new_c));
    }
}
