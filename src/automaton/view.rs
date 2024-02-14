use portgraph::PortView;

use super::{OutPort, ScopeAutomaton, StateID, Transition};
use crate::{
    constraint::{Scope, ScopeConstraint},
    PatternID,
};

impl<C: ScopeConstraint + Clone> ScopeAutomaton<C> {
    pub(super) fn outputs(&self, state: StateID) -> impl Iterator<Item = OutPort> + '_ {
        self.graph.outputs(state.0).map(move |p| {
            let offset = self.graph.port_offset(p).expect("invalid port").index();
            OutPort(state, offset)
        })
    }

    #[allow(unused)]
    pub(crate) fn n_states(&self) -> usize {
        self.graph.node_count()
    }

    pub(super) fn predicate(&self, edge: OutPort) -> &Transition<C> {
        let OutPort(state, offset) = edge;
        let port = self.graph.output(state.0, offset).unwrap();
        self.weights[port].as_ref().expect("Invalid outgoing port transition")
    }

    pub(super) fn scope(&self, state: StateID) -> &Scope<C> {
        &self.weights[state.0].as_ref().expect("invalid state").scope
    }

    pub(super) fn matches(&self, state: StateID) -> impl Iterator<Item = PatternID> + '_
    where
        C: Clone,
    {
        self.weights[state.0]
            .as_ref()
            .expect("invalid state")
            .matches
            .iter()
            .copied()
    }

    pub(super) fn is_deterministic(&self, state: StateID) -> bool {
        self.weights[state.0]
            .as_ref()
            .expect("invalid state")
            .deterministic
    }

    #[allow(unused)]
    pub(crate) fn states(&self) -> impl Iterator<Item = StateID> + '_ {
        self.graph.nodes_iter().map(StateID)
    }
}
