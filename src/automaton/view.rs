use petgraph::{graph::NodeIndex, visit::IntoNodeIdentifiers, Graph};

use crate::{EdgeProperty, HashSet, PatternID};

use super::{EdgePredicate, OutPort, ScopeAutomaton, StateID, Symbol};

impl<PNode: Clone, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    pub(super) fn out_ports(&self, StateID(state): StateID) -> impl Iterator<Item = OutPort> + '_ {
        let n_out = graph_node_weight(&self.graph, state).predicates.len();
        (0..n_out).map(move |position| OutPort { state, position })
    }

    pub(super) fn any_out_ports(&self, state: StateID) -> bool {
        self.out_ports(state).any(|_| true)
    }

    #[allow(unused)]
    pub(crate) fn n_states(&self) -> usize {
        self.graph.node_count()
    }

    pub(super) fn predicate(
        &self,
        out_port: OutPort,
    ) -> &EdgePredicate<PNode, PEdge, PEdge::OffsetID> {
        let predicates = &graph_node_weight(&self.graph, out_port.state).predicates;
        &predicates[out_port.position].0
    }

    pub(super) fn scope(&self, StateID(state): StateID) -> &HashSet<Symbol> {
        &graph_node_weight(&self.graph, state).scope
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

    #[allow(unused)]
    pub(crate) fn states(&self) -> impl Iterator<Item = StateID> + '_ {
        self.graph.node_identifiers().map(StateID)
    }

    /// Follow edge from an OutPort to the next state
    pub(super) fn next_state(&self, out_port: OutPort) -> StateID {
        let edge = graph_node_weight(&self.graph, out_port.state).predicates[out_port.position].1;
        self.graph
            .edge_endpoints(edge)
            .expect("invalid edge")
            .1
            .into()
    }
}

pub(super) fn graph_node_weight<N, E>(graph: &Graph<Option<N>, E>, state: NodeIndex) -> &N {
    graph
        .node_weight(state)
        .expect("unknown state")
        .as_ref()
        .expect("invalid state")
}
