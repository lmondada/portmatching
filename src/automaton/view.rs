use petgraph::{graph::NodeIndex, stable_graph::StableGraph, visit::EdgeRef, Direction};

use crate::PatternID;

use super::{ScopeAutomaton, StateID, TransitionID};

impl<C> ScopeAutomaton<C> {
    pub(super) fn transitions(
        &self,
        StateID(state): StateID,
    ) -> impl Iterator<Item = TransitionID> + '_ {
        self.graph
            .edges_directed(state, Direction::Outgoing)
            .map(|e| e.id().into())
    }

    pub(super) fn constraint(&self, transition: TransitionID) -> Option<&C> {
        self.graph[transition.0].constraint.as_ref()
    }

    pub(super) fn children(&self, state: StateID) -> impl Iterator<Item = StateID> + '_ {
        self.transitions(state)
            .map(|transition| self.next_state(transition))
    }

    /// All non-None constraints at `state`.
    pub(super) fn constraints(&self, state: StateID) -> impl Iterator<Item = &C> + '_ {
        self.transitions(state)
            .filter_map(|transition| self.constraint(transition))
    }

    pub(super) fn matches(&self, StateID(state): StateID) -> impl Iterator<Item = PatternID> + '_ {
        graph_node_weight(&self.graph, state)
            .matches
            .iter()
            .copied()
    }

    pub(super) fn is_deterministic(&self, StateID(state): StateID) -> bool {
        graph_node_weight(&self.graph, state).deterministic
    }

    /// Follow edge from an OutPort to the next state
    pub(super) fn next_state(&self, transition: TransitionID) -> StateID {
        self.graph
            .edge_endpoints(transition.0)
            .expect("invalid transition")
            .1
            .into()
    }

    pub(super) fn find_constraint(
        &self,
        state: StateID,
        constraint: Option<&C>,
    ) -> Option<TransitionID>
    where
        C: Eq,
    {
        self.transitions(state)
            .find(|&transition| self.constraint(transition) == constraint)
    }
}

pub(super) fn graph_node_weight<N, E>(graph: &StableGraph<N, E>, state: NodeIndex) -> &N {
    graph.node_weight(state).expect("unknown state")
}
