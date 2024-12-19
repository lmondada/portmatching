use std::collections::BTreeSet;

use petgraph::{algo::toposort, visit::EdgeRef, Direction};

use crate::{indexing::IndexKey, HashMap, PatternID};

use super::{ConstraintAutomaton, State, StateID, TransitionGraph, TransitionID};

/// A simple internal trait for accessing the underlying graph
pub(super) trait GraphView<K: IndexKey, B> {
    /// A reference to the underlying graph
    fn underlying_graph(&self) -> &TransitionGraph<K, B>;

    /// Get the next state obtained from following a transition
    fn next_state(&self, transition: TransitionID) -> StateID {
        self.underlying_graph()
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .1
            .into()
    }

    fn incoming_transitions<'a>(
        &'a self,
        StateID(state): StateID,
    ) -> impl Iterator<Item = TransitionID> + 'a
    where
        B: 'a,
    {
        self.underlying_graph()
            .edges_directed(state, Direction::Incoming)
            .map(|e| e.id().into())
    }

    fn matches<'a>(&'a self, state: StateID) -> &'a HashMap<PatternID, Vec<K>>
    where
        B: 'a,
    {
        &self.node_weight(state).matches
    }

    /// The start state of a transition
    fn parent(&self, transition: TransitionID) -> StateID {
        self.underlying_graph()
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .0
            .into()
    }

    fn parents<'a>(&'a self, state: StateID) -> impl Iterator<Item = StateID> + 'a
    where
        B: 'a,
    {
        self.incoming_transitions(state)
            .map(|transition| self.parent(transition))
    }

    fn branch_selector(&self, state: StateID) -> Option<&B> {
        self.node_weight(state).branch_selector.as_ref()
    }

    fn all_states(&self) -> impl Iterator<Item = StateID> {
        let nodes = toposort(self.underlying_graph(), Default::default()).unwrap();
        nodes.into_iter().map(StateID)
    }

    fn min_scope<'a>(&'a self, state: StateID) -> &'a [K]
    where
        B: 'a,
    {
        &self.node_weight(state).min_scope
    }

    fn max_scope<'a>(&'a self, state: StateID) -> &'a BTreeSet<K>
    where
        B: 'a,
    {
        &self.node_weight(state).max_scope
    }

    fn node_weight<'a>(&'a self, state: StateID) -> &'a State<K, B>
    where
        B: 'a,
    {
        self.underlying_graph()
            .node_weight(state.0)
            .expect("unknown state")
    }

    /// All non-Fail constraints at `state`.
    fn all_constraint_transitions<'a>(
        &'a self,
        state: StateID,
    ) -> impl Iterator<Item = TransitionID> + 'a
    where
        B: 'a,
    {
        self.node_weight(state).edge_order.iter().copied()
    }

    /// The fail transition at `state`, if it exists
    fn epsilon_transition(&self, state: StateID) -> Option<TransitionID> {
        self.node_weight(state).epsilon
    }

    /// The state reached by the fail transition at `state`, if any.
    fn fail_child(&self, state: StateID) -> Option<StateID> {
        self.epsilon_transition(state)
            .map(|transition| self.next_state(transition))
    }

    fn nth_child(&self, state: StateID, n: usize) -> StateID {
        let transition = self.node_weight(state).edge_order[n];
        self.next_state(transition)
    }

    /// Iterate over all transitions of a state, in order.
    fn all_transitions<'a>(&'a self, state: StateID) -> impl Iterator<Item = TransitionID> + 'a
    where
        B: 'a,
    {
        self.all_constraint_transitions(state)
            .chain(self.epsilon_transition(state))
    }

    /// The states reached by a single transition from `state`
    #[allow(dead_code)]
    fn children<'a>(&'a self, state: StateID) -> impl Iterator<Item = StateID> + 'a
    where
        B: 'a,
    {
        self.all_transitions(state)
            .map(|transition| self.next_state(transition))
    }

    // /// Get a tuple of data that fully specifies the action of a `state`.
    // ///
    // /// Two states with the same tuple ID may be merged into one without
    // /// changing the behavior of the matcher.
    // fn state_tuple(&self, state: StateID) -> StateTuple<Constraint<K, B>> {
    //     let transitions = self
    //         .all_transitions(state)
    //         .map(|transition| (self.constraint(transition), self.next_state(transition)))
    //         .collect();
    //     StateTuple {
    //         deterministic: self.is_deterministic(state),
    //         matches: self.matches(state).keys().copied().collect(),
    //         transitions,
    //     }
    // }
}

impl<K: IndexKey, B> GraphView<K, B> for ConstraintAutomaton<K, B> {
    fn underlying_graph(&self) -> &TransitionGraph<K, B> {
        &self.graph
    }
}
