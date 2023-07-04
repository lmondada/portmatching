use portgraph::{LinkMut, PortMut, PortView};

use crate::{predicate::EdgePredicate, PatternID};

use super::{ScopeAutomaton, State, StateID, Symbol, Transition};

impl<PNode: Copy, PEdge: Copy> ScopeAutomaton<PNode, PEdge> {
    pub(super) fn set_children(
        &mut self,
        state: StateID,
        preds: &[EdgePredicate<PNode, PEdge, Symbol>],
        next_states: &[Option<StateID>],
    ) -> Vec<Option<StateID>> {
        if self.graph.num_outputs(state.0) != 0 {
            panic!("State already has outgoing ports");
        }
        // Allocate new ports
        self.add_ports(state, 0, preds.len());

        // Build the children
        preds
            .iter()
            .zip(next_states)
            .enumerate()
            .map(|(i, (&pred, &next_state))| self.add_child(state, i, pred.into(), next_state))
            .collect()
    }

    fn add_child(
        &mut self,
        parent: StateID,
        offset: usize,
        pedge: Transition<PNode, PEdge>,
        new_state: Option<StateID>,
    ) -> Option<StateID> {
        let mut added_state = false;
        let (new_state, new_offset) = if let Some(new_state) = new_state {
            let in_offset = self.graph.num_inputs(new_state.0);
            self.add_ports(new_state, 1, 0);
            (new_state, in_offset)
        } else {
            added_state = true;
            (self.graph.add_node(1, 0).into(), 0)
        };
        self.graph
            .link_nodes(parent.0, offset, new_state.0, new_offset)
            .expect("Could not add child at offset p");
        let new_scope = {
            let mut old_scope = self.weights[parent.0]
                .clone()
                .expect("invalid parent")
                .scope;
            if let EdgePredicate::LinkNewNode { new_node, .. } = pedge.into() {
                old_scope.insert(new_node);
            }
            old_scope
        };
        self.weights.nodes[new_state.0] = Some(State {
            matches: Vec::new(),
            scope: new_scope,
            deterministic: true,
        });
        self.weights[self.graph.output(parent.0, offset).unwrap()] = Some(pedge.into());
        added_state.then_some(new_state)
    }

    fn add_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        let incoming = incoming + self.graph.num_inputs(state.0);
        let outgoing = outgoing + self.graph.num_outputs(state.0);
        self.graph
            .set_num_ports(state.0, incoming, outgoing, |old, new| {
                self.weights.ports.rekey(old, new.new_index());
            });
    }

    pub(crate) fn add_match(&mut self, state: StateID, pattern: PatternID) {
        self.weights[state.0]
            .as_mut()
            .expect("invalid state")
            .matches
            .push(pattern);
    }

    pub(crate) fn to_non_det(&mut self, state: StateID) {
        if self.graph.num_outputs(state.0) > 0 {
            panic!("Cannot make state non-deterministic: has outgoing ports");
        }
        self.weights[state.0]
            .as_mut()
            .expect("invalid state")
            .deterministic = false;
    }
}
