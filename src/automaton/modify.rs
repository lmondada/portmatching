use itertools::izip;
use portgraph::{LinkMut, PortMut, PortView};

use crate::{constraint::ScopeConstraint, HashSet, PatternID};

use super::{ScopeAutomaton, State, StateID, Transition};

impl<C: ScopeConstraint + Clone> ScopeAutomaton<C> {
    pub(super) fn set_children(
        &mut self,
        state: StateID,
        transitions: impl Iterator<Item = Transition<C>> + ExactSizeIterator,
        next_states: &[Option<StateID>],
        next_scopes: Vec<HashSet<C::Symbol>>,
    ) -> Vec<Option<StateID>> {
        if self.graph.num_outputs(state.0) != 0 {
            panic!("State already has outgoing ports");
        }
        // Allocate new ports
        self.add_ports(state, 0, transitions.len());

        // Build the children
        izip!(transitions, next_states, next_scopes)
            .enumerate()
            .map(|(i, (t, &next_state, next_scope))| {
                self.add_child(state, i, t.into(), next_state, next_scope)
            })
            .collect()
    }

    fn add_child(
        &mut self,
        parent: StateID,
        offset: usize,
        transition: Transition<C>,
        new_state: Option<StateID>,
        new_scope: HashSet<C::Symbol>,
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
        self.weights[self.graph.output(parent.0, offset).unwrap()] = Some(transition);
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
    // use super::*;

    // /// The child state's scope should be the intersection of all possible scopes
    // TODO: #[test]
    // fn intersect_scope() {
    //     let mut a = ScopeAutomaton::new();
    //     a.add_ports(a.root(), 0, 2);
    //     let s_root = Symbol::root();
    //     let s1 = Symbol::new(IterationStatus::Finished, 0);
    //     let s2 = Symbol::new(IterationStatus::Finished, 1);
    //     let t1: Transition<(), (), ()> = EdgePredicate::LinkNewNode {
    //         node: s_root,
    //         property: (),
    //         new_node: s1,
    //     }
    //     .into();
    //     let t2: Transition<(), (), ()> = EdgePredicate::LinkNewNode {
    //         node: s_root,
    //         property: (),
    //         new_node: s2,
    //     }
    //     .into();
    //     let child = a.add_child(a.root(), 0, t1, None, None).unwrap();

    //     assert_eq!(a.scope(child), &[s_root, s1].into_iter().collect());
    //     a.add_child(a.root(), 1, t2, Some(child), None);
    //     assert_eq!(a.scope(child), &[s_root].into_iter().collect());
    // }
}
