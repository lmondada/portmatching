use petgraph::{visit::EdgeRef, Direction};

use crate::{HashSet, PatternID};

use super::{ConstraintAutomaton, State, StateID, TransitionID};

/// Methods for viewing the automaton
///
/// Exposed as a trait so that the automaton builder can reuse the default
/// implementation but trace calls where useful.
impl<C: Eq, I> ConstraintAutomaton<C, I> {
    /// Find the transition ID at `parent` with the given `constraint`
    pub(super) fn find_constraint(
        &self,
        state: StateID,
        constraint: Option<&C>,
    ) -> Option<TransitionID> {
        self.transitions(state)
            .find(|&transition| self.constraint(transition) == constraint)
    }

    pub(super) fn find_fail_transition(&self, state: StateID) -> Option<TransitionID> {
        self.transitions(state)
            .find(|&transition| self.constraint(transition).is_none())
    }

    /// Get the next state obtained from following a transition
    pub(super) fn next_state(&self, transition: TransitionID) -> StateID {
        self.graph
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .1
            .into()
    }

    /// Iterate over all transitions of a state, in order.
    pub(super) fn transitions(&self, state: StateID) -> impl Iterator<Item = TransitionID> + '_ {
        self.node_weight(state).order.iter().copied()
    }

    /// Get the constraint corresponding to a transition
    pub(super) fn constraint(&self, transition: TransitionID) -> Option<&C> {
        self.graph[transition.0].constraint.as_ref()
    }

    /// The states reached by a single transition from `state`
    #[allow(dead_code)]
    pub(super) fn children(&self, state: StateID) -> impl Iterator<Item = StateID> + '_ {
        self.transitions(state)
            .map(|transition| self.next_state(transition))
    }

    pub(super) fn incoming_transitions(
        &self,
        StateID(state): StateID,
    ) -> impl Iterator<Item = TransitionID> + '_ {
        self.graph
            .edges_directed(state, Direction::Incoming)
            .map(|e| e.id().into())
    }

    /// All non-None constraints at `state`.
    pub(super) fn constraints(&self, state: StateID) -> impl Iterator<Item = &C> + '_ {
        self.transitions(state)
            .filter_map(|transition| self.constraint(transition))
    }

    pub(super) fn matches(&self, state: StateID) -> &[PatternID] {
        &self.node_weight(state).matches
    }

    pub(super) fn is_deterministic(&self, state: StateID) -> bool {
        self.node_weight(state).deterministic
    }

    /// The start state of a transition
    pub(super) fn parent(&self, transition: TransitionID) -> StateID {
        self.graph
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .0
            .into()
    }
}

// Small, private utils functions
impl<C, I> ConstraintAutomaton<C, I> {
    pub(super) fn node_weight(&self, state: StateID) -> &State {
        self.graph.node_weight(state.0).expect("unknown state")
    }

    #[allow(dead_code)]
    pub(super) fn check_edge_order_invariant(&self, state: StateID) -> bool {
        self.node_weight(state)
            .order
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            == self
                .graph
                .edges(state.0)
                .map(|e| TransitionID(e.id()))
                .collect::<HashSet<_>>()
    }
}
