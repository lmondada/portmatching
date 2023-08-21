use portgraph::{LinkMut, PortMut, PortView};

use crate::{predicate::EdgePredicate, EdgeProperty, PatternID};

use super::{ScopeAutomaton, State, StateID, Transition};

impl<PNode: Clone, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    pub(super) fn set_children<I>(
        &mut self,
        state: StateID,
        preds: impl IntoIterator<IntoIter = I>,
        next_states: &[Option<StateID>],
    ) -> Vec<Option<StateID>>
    where
        I: Iterator<Item = EdgePredicate<PNode, PEdge, PEdge::OffsetID>> + ExactSizeIterator,
    {
        let preds = preds.into_iter();
        if self.graph.num_outputs(state.0) != 0 {
            panic!("State already has outgoing ports");
        }
        // Allocate new ports
        self.add_ports(state, 0, preds.len());

        // Build the children
        preds
            .zip(next_states)
            .enumerate()
            .map(|(i, (pred, &next_state))| self.add_child(state, i, pred.into(), next_state))
            .collect()
    }

    fn add_child(
        &mut self,
        parent: StateID,
        offset: usize,
        pedge: Transition<PNode, PEdge, PEdge::OffsetID>,
        new_state: Option<StateID>,
    ) -> Option<StateID> {
        let mut added_state = false;
        let (new_state_id, new_offset) = if let Some(new_state) = new_state {
            let in_offset = self.graph.num_inputs(new_state.0);
            self.add_ports(new_state, 1, 0);
            (new_state, in_offset)
        } else {
            added_state = true;
            (self.graph.add_node(1, 0).into(), 0)
        };
        self.graph
            .link_nodes(parent.0, offset, new_state_id.0, new_offset)
            .expect("Could not add child at offset p");
        let new_scope = {
            let mut old_scope = self.weights[parent.0]
                .clone()
                .expect("invalid parent")
                .scope;
            if let EdgePredicate::LinkNewNode { new_node, .. } = pedge.clone().into() {
                old_scope.insert(new_node);
            }
            old_scope
        };
        let new_state = if let Some(mut new_state) = self.weights[new_state_id.0].take() {
            new_state.scope.retain(|k| new_scope.contains(k));
            new_state
        } else {
            State {
                matches: Vec::new(),
                scope: new_scope,
                deterministic: true,
            }
        };
        self.weights.nodes[new_state_id.0] = Some(new_state);
        self.weights[self.graph.output(parent.0, offset).unwrap()] = Some(pedge);
        added_state.then_some(new_state_id)
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

    pub(crate) fn make_non_det(&mut self, state: StateID) {
        if self.graph.num_outputs(state.0) > 0 {
            panic!("Cannot make state non-deterministic: has outgoing ports");
        }
        self.weights[state.0]
            .as_mut()
            .expect("invalid state")
            .deterministic = false;
    }
}

#[cfg(test)]
mod tests {
    use crate::{patterns::IterationStatus, predicate::Symbol};

    use super::*;

    /// The child state's scope should be the intersection of all possible scopes
    #[test]
    fn intersect_scope() {
        let mut a = ScopeAutomaton::new();
        a.add_ports(a.root(), 0, 2);
        let s_root = Symbol::root();
        let s1 = Symbol::new(IterationStatus::Finished, 0);
        let s2 = Symbol::new(IterationStatus::Finished, 1);
        let t1: Transition<(), (), ()> = EdgePredicate::LinkNewNode {
            node: s_root,
            property: (),
            new_node: s1,
        }
        .into();
        let t2: Transition<(), (), ()> = EdgePredicate::LinkNewNode {
            node: s_root,
            property: (),
            new_node: s2,
        }
        .into();
        let child = a.add_child(a.root(), 0, t1, None).unwrap();

        assert_eq!(a.scope(child), &[s_root, s1].into_iter().collect());
        a.add_child(a.root(), 1, t2, Some(child));
        assert_eq!(a.scope(child), &[s_root].into_iter().collect());
    }
}
