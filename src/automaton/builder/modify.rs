use std::collections::BTreeSet;

use petgraph::data::{Build, DataMapMut};

use crate::automaton::view::GraphView;
use crate::automaton::{State, StateID, TransitionGraph, TransitionID};
use crate::indexing::IndexKey;
use crate::PatternID;

use super::AutomatonBuilder;

impl<P, K: IndexKey, B> GraphView<K, B> for AutomatonBuilder<P, K, B> {
    fn underlying_graph(&self) -> &TransitionGraph<K, B> {
        self.graph.inner()
    }
}

/// Methods for modifying the automaton
impl<P, K: IndexKey, B> AutomatonBuilder<P, K, B> {
    /// Add a child state, record matches and link to parent
    ///
    /// If the child state already exists, it will be reused and `None` will
    /// be returned. Otherwise, the new state will be returned
    pub(super) fn add_child(
        &mut self,
        parent: StateID,
        patterns: BTreeSet<(PatternID, P)>,
        matches: BTreeSet<PatternID>,
        is_fail: bool,
    ) -> Option<StateID>
    where
        P: Clone + Ord,
    {
        let hash_key = (patterns.clone(), matches.clone());

        let mut child_created = false;

        // First create (or reuse) the child state
        let mut child = None;
        if let Some(&c) = self.hashcons.get(&hash_key) {
            // Requirement for reusing the child: parent -> c is a valid edge
            if self.graph.is_valid_edge(parent.0, c.0) {
                child = Some(c);
            }
        }
        let child = child.unwrap_or_else(|| {
            let node = self.graph.add_node(State::default());
            let state = StateID(node);
            // The newest state is always the best to hashcons to
            self.hashcons.insert(hash_key, state);
            child_created = true;
            state
        });

        // Now add the parent to child edge
        let edge_id = self.graph.try_add_edge(parent.0, child.0, ()).unwrap();
        let tr = TransitionID(edge_id);

        // Record the edge in the state order
        if is_fail {
            let epsilon = &mut self.node_weight_mut(parent).epsilon;
            assert!(epsilon.is_none());
            *epsilon = Some(tr);
        } else {
            let order = &mut self.node_weight_mut(parent).edge_order;
            order.push(tr);
        }

        // Finally, add the matches
        self.add_matches(child, matches.iter().copied());

        child_created.then_some(child)
    }

    pub(super) fn set_max_scope(&mut self, state: StateID, max_scope: BTreeSet<K>) {
        let weight = self.node_weight_mut(state);
        weight.max_scope = max_scope;
    }

    pub(super) fn set_min_scope(&mut self, state: StateID, min_scope: Vec<K>) {
        let weight = self.node_weight_mut(state);
        weight.min_scope = min_scope;
    }

    pub(super) fn set_branch_selector(&mut self, state: StateID, branch_selector: B) {
        let weight = self.node_weight_mut(state);
        weight.branch_selector = Some(branch_selector);
    }

    pub(super) fn add_matches(
        &mut self,
        state: StateID,
        patterns: impl IntoIterator<Item = PatternID>,
    ) where
        K: IndexKey,
    {
        for id in patterns {
            let scope = Vec::from_iter(self.get_scope(id).iter().copied());
            self.node_weight_mut(state).matches.insert(id, scope);
        }
    }
}

// Small, private utils functions
impl<P, K: Ord, B> AutomatonBuilder<P, K, B> {
    fn node_weight_mut(&mut self, StateID(state): StateID) -> &mut State<K, B> {
        self.graph.node_weight_mut(state).expect("invalid state")
    }
}

#[cfg(test)]
pub mod tests {
    use itertools::Itertools;
    use petgraph::acyclic::Acyclic;
    use rstest::{fixture, rstest};

    use crate::{
        automaton::{builder::tests::TestBuildConfig, tests::TestAutomaton, AutomatonBuilder},
        branch_selector::tests::TestBranchSelector,
        constraint::tests::TestConstraint,
        predicate::tests::{TestKey, TestPattern, TestPredicate},
    };

    use super::*;

    pub type TestAutomatonBuilder = AutomatonBuilder<TestPattern, TestKey, TestBranchSelector>;

    impl TestAutomatonBuilder {
        pub(crate) fn from_matcher(automaton: TestAutomaton) -> TestAutomatonBuilder {
            let TestAutomaton { graph, root } = automaton;
            Self {
                graph: Acyclic::try_from_graph(graph).unwrap(),
                root,
                ..Default::default()
            }
        }
    }
}
