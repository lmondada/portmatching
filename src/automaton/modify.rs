use itertools::izip;
use petgraph::{graph::NodeIndex, Graph};

use crate::{
    predicate::{EdgePredicate, Symbol},
    EdgeProperty, HashSet, PatternID,
};

use super::{view::graph_node_weight, ScopeAutomaton, State, StateID};

impl<PNode: Clone, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    pub(super) fn set_children<I>(
        &mut self,
        state: StateID,
        preds: impl IntoIterator<IntoIter = I>,
        next_states: &[Option<StateID>],
        next_scopes: Vec<HashSet<Symbol>>,
    ) -> Vec<Option<StateID>>
    where
        I: Iterator<Item = EdgePredicate<PNode, PEdge, PEdge::OffsetID>> + ExactSizeIterator,
    {
        let preds = preds.into_iter();
        if self.any_out_ports(state) {
            panic!("State already has outgoing ports");
        }
        // Build the children
        izip!(preds, next_states, next_scopes)
            .enumerate()
            .map(|(i, (pred, &next_state, next_scope))| {
                self.add_transition(state, i, pred.into(), next_state, Some(next_scope))
            })
            .collect()
    }

    fn add_transition(
        &mut self,
        StateID(parent): StateID,
        position: usize,
        predicate: EdgePredicate<PNode, PEdge, PEdge::OffsetID>,
        new_state: Option<StateID>,
        new_scope: Option<HashSet<Symbol>>,
    ) -> Option<StateID> {
        // Create state if it does not exist
        let mut added_state = false;
        let new_state = match new_state {
            Some(StateID(new_state)) => new_state,
            None => {
                added_state = true;
                self.graph.add_node(None).into()
            }
        };

        // Add edge to new state
        let new_edge = self.graph.add_edge(parent, new_state, ());

        // Update scope of new state
        let new_scope = new_scope.unwrap_or_else(|| {
            // By default, take scope of parent and add symbol if necessary
            let mut old_scope = graph_node_weight(&mut self.graph, parent).scope.clone();
            if let EdgePredicate::LinkNewNode { new_node, .. } = predicate {
                old_scope.insert(new_node);
            }
            old_scope
        });

        let new_state_weight = self.graph.node_weight_mut(new_state).unwrap();
        if let Some(new_state_weight) = new_state_weight {
            new_state_weight.scope.retain(|k| new_scope.contains(k));
        } else {
            *new_state_weight = Some(State {
                matches: vec![],
                predicates: vec![],
                scope: new_scope,
                deterministic: true,
            });
        }

        // Finally, add predicate to parent at `position`
        let parent_weight = graph_node_weight_mut(&mut self.graph, parent);
        parent_weight
            .predicates
            .insert(position, (predicate, new_edge));

        added_state.then_some(StateID(new_state))
    }

    pub(crate) fn add_match(&mut self, StateID(state): StateID, pattern: PatternID) {
        graph_node_weight_mut(&mut self.graph, state)
            .matches
            .push(pattern);
    }

    pub(crate) fn make_non_det(&mut self, state: StateID) {
        if self.any_out_ports(state) {
            panic!("Cannot make state non-deterministic: has outgoing ports");
        }
        graph_node_weight_mut(&mut self.graph, state.0).deterministic = false;
    }
}

fn graph_node_weight_mut<N, E>(graph: &mut Graph<Option<N>, E>, state: NodeIndex) -> &mut N {
    graph
        .node_weight_mut(state)
        .expect("unknown state")
        .as_mut()
        .expect("invalid state")
}

#[cfg(test)]
mod tests {
    use crate::{patterns::IterationStatus, predicate::Symbol};

    use super::*;

    /// The child state's scope should be the intersection of all possible scopes
    #[test]
    fn intersect_scope() {
        let mut a = ScopeAutomaton::new();
        let s_root = Symbol::root();
        let s1 = Symbol::new(IterationStatus::Finished, 0);
        let s2 = Symbol::new(IterationStatus::Finished, 1);
        let t1: EdgePredicate<(), (), ()> = EdgePredicate::LinkNewNode {
            node: s_root,
            property: (),
            new_node: s1,
        }
        .into();
        let t2 = EdgePredicate::LinkNewNode {
            node: s_root,
            property: (),
            new_node: s2,
        }
        .into();
        let child = a.add_transition(a.root(), 0, t1, None, None).unwrap();

        assert_eq!(a.scope(child), &[s_root, s1].into_iter().collect());
        a.add_transition(a.root(), 1, t2, Some(child), None);
        assert_eq!(a.scope(child), &[s_root].into_iter().collect());
    }
}
