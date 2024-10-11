use petgraph::{visit::EdgeRef, Direction};

use crate::{indexing::IndexKey, Constraint, HashMap, HashSet, PatternID};

use super::{ConstraintAutomaton, State, StateID, TransitionID};

/// Methods for viewing the automaton
///
/// Exposed as a trait so that the automaton builder can reuse the default
/// implementation but trace calls where useful.
impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I>
where
    Constraint<K, P>: Eq,
{
    /// All non-None constraints at `state`.
    pub(super) fn all_constraint_transitions(
        &self,
        state: StateID,
    ) -> impl Iterator<Item = TransitionID> + '_ {
        self.node_weight(state).constraint_order.iter().copied()
    }

    /// All None constraints at `state`.
    pub(super) fn all_epsilon_transitions(
        &self,
        state: StateID,
    ) -> impl ExactSizeIterator<Item = TransitionID> + '_ {
        self.node_weight(state).epsilon_order.iter().copied()
    }

    /// The state reached by the fail transition at `state`, if any.
    ///
    /// Expects at most one epsilon transition, otherwise panics.
    pub(super) fn fail_next_state(&self, state: StateID) -> Option<StateID> {
        assert!(self.all_epsilon_transitions(state).len() <= 1);
        self.all_epsilon_transitions(state)
            .next()
            .map(|transition| self.next_state(transition))
    }

    /// The state reached by the `constraint` transition at `state`, if any.
    ///
    /// Expects at most one transition, otherwise panics.
    #[allow(dead_code)]
    pub(super) fn constraint_next_state(
        &self,
        state: StateID,
        constraint: &Constraint<K, P>,
    ) -> Option<StateID> {
        self.all_constraint_transitions(state)
            .find(|&t| self.constraint(t) == Some(constraint))
            .map(|t| self.next_state(t))
    }

    /// Iterate over all transitions of a state, in order.
    pub(super) fn all_transitions(
        &self,
        state: StateID,
    ) -> impl Iterator<Item = TransitionID> + '_ {
        self.all_constraint_transitions(state)
            .chain(self.all_epsilon_transitions(state))
    }

    pub(super) fn constraints(
        &self,
        state: StateID,
    ) -> impl Iterator<Item = &Constraint<K, P>> + '_ {
        self.all_constraint_transitions(state)
            .map(|transition| self.constraint(transition).unwrap())
    }

    /// The states reached by a single transition from `state`
    #[allow(dead_code)]
    pub(super) fn children(&self, state: StateID) -> impl Iterator<Item = StateID> + '_ {
        self.all_transitions(state)
            .map(|transition| self.next_state(transition))
    }

    /// Get a tuple of data that fully specifies the action of a `state`.
    ///
    /// Two states with the same tuple ID may be merged into one without
    /// changing the behavior of the matcher.
    pub(super) fn state_tuple(&self, state: StateID) -> StateTuple<Constraint<K, P>> {
        let transitions = self
            .all_transitions(state)
            .map(|transition| (self.constraint(transition), self.next_state(transition)))
            .collect();
        StateTuple {
            deterministic: self.is_deterministic(state),
            matches: self.matches(state).keys().copied().collect(),
            transitions,
        }
    }
}

impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I> {
    /// Get the next state obtained from following a transition
    pub(super) fn next_state(&self, transition: TransitionID) -> StateID {
        self.graph
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .1
            .into()
    }

    pub(super) fn is_unreachable(&self, state: StateID) -> bool {
        self.incoming_transitions(state).next().is_none()
    }

    /// Get the constraint corresponding to a transition
    pub(super) fn constraint(&self, transition: TransitionID) -> Option<&Constraint<K, P>> {
        self.graph[transition.0].constraint.as_ref()
    }

    pub(super) fn incoming_transitions(
        &self,
        StateID(state): StateID,
    ) -> impl Iterator<Item = TransitionID> + '_ {
        self.graph
            .edges_directed(state, Direction::Incoming)
            .map(|e| e.id().into())
    }

    pub(super) fn matches(&self, state: StateID) -> &HashMap<PatternID, HashSet<K>> {
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

    pub(super) fn required_bindings(&self, state: StateID) -> &[K] {
        &self.node_weight(state).required_bindings
    }
}

/// Some data that fully specifies the action of a state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct StateTuple<'a, C> {
    deterministic: bool,
    matches: Vec<PatternID>,
    transitions: Vec<(Option<&'a C>, StateID)>,
}

// Small, private utils functions
impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I> {
    pub(super) fn node_weight(&self, state: StateID) -> &State<K> {
        self.graph.node_weight(state.0).expect("unknown state")
    }
}
