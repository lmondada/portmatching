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
    /// Add a new disconnected non-deterministic state
    pub(super) fn add_non_det_node(&mut self) -> StateID {
        let node = self.graph.add_node(State::default());
        StateID(node)
    }

    /// Set the deterministic flag of `state` to true and return whether it was
    /// already deterministic.
    pub(super) fn set_deterministic(&mut self, state: StateID) -> bool {
        let node = self.node_weight_mut(state);
        let was_det = node.deterministic;
        node.deterministic = true;
        was_det
    }

    pub(super) fn add_child(
        &mut self,
        parent: StateID,
        patterns: BTreeSet<(PatternID, P)>,
        is_fail: bool,
    ) -> StateID
    where
        P: Ord,
    {
        assert!(
            self.branch_selector(parent).is_some(),
            "cannot add child to non-branching state"
        );

        // First create (or reuse) the child state
        let mut child = None;
        if let Some(&c) = self.hashcons.get(&patterns) {
            // Requirement for reusing the child: parent -> c is a valid edge
            if self.graph.is_valid_edge(parent.0, c.0) {
                child = Some(c);
            }
        }
        let child = child.unwrap_or_else(|| {
            let node = self.graph.add_node(State::default());
            let state = StateID(node);
            // The newest state is always the best to hashcons to
            self.hashcons.insert(patterns, state);
            state
        });

        // Now add the parent to child edge
        let edge_id = self.graph.try_add_edge(parent.0, child.0, ()).unwrap();
        let tr = TransitionID(edge_id);

        // Record the edge in the state order
        if is_fail {
            let epsilon = &mut self.node_weight_mut(child).epsilon;
            assert!(epsilon.is_none());
            *epsilon = Some(tr);
        } else {
            let order = &mut self.node_weight_mut(parent).edge_order;
            order.push(tr);
        }

        child
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

    /// An automaton with a X transition at the root and transitions
    /// [a,b,c,d] at the only child
    #[fixture]
    pub fn automaton() -> TestAutomaton {
        let [constraint_root, constraint_a, constraint_b] = constraints();
        let p1 = TestPattern::from_constraints([constraint_root.clone(), constraint_a.clone()]);
        let p2 = TestPattern::from_constraints([constraint_root.clone(), constraint_b.clone()]);
        let builder = AutomatonBuilder::from_patterns(vec![p1, p2]);
        builder.build(TestBuildConfig::default()).0
    }

    #[fixture]
    pub fn automaton2() -> TestAutomaton {
        let automaton = automaton();
        let x_child = root_child(&automaton);
        let [a_child, _, _, _] = root_grandchildren(&automaton);

        let builder = AutomatonBuilder::from_matcher(automaton);

        todo!();
        // // Add a FAIL transition from x_child to a new state
        // let fail_child = builder.add_child(x_child, None, true);
        // // Add a common constraint to the fail child and a_child
        // let common_constraint = TestConstraint::new(vec![7, 8]);
        // let post_fail = builder.add_constraint(fail_child, common_constraint.clone());
        // builder.add_constraint(a_child, common_constraint.clone());
        // // Add a second common constraint to post_fail
        // let common_constraint2 = TestConstraint::new(vec![77, 8]);
        // builder.add_constraint(post_fail, common_constraint2.clone());
        builder.build(TestBuildConfig::default()).0
    }

    pub fn constraints() -> [TestConstraint; 3] {
        [
            TestConstraint::new(TestPredicate::AreEqual),
            TestConstraint::new(TestPredicate::NotEqual),
            TestConstraint::new(TestPredicate::PredTwo),
        ]
    }

    pub(crate) fn root_child(automaton: &TestAutomaton) -> StateID {
        let cs = automaton.children(automaton.root()).collect_vec();
        assert_eq!(cs.len(), 1);
        cs[0]
    }

    pub(crate) fn root_grandchildren(automaton: &TestAutomaton) -> [StateID; 4] {
        let (child_a, child_b, child_c, child_d) = automaton
            .all_constraint_transitions(root_child(automaton))
            .map(|t| automaton.next_state(t))
            .collect_tuple()
            .unwrap();
        [child_a, child_b, child_c, child_d]
    }

    #[rstest]
    fn test_add_constraints(automaton: TestAutomaton) {
        assert_eq!(automaton.graph.node_count(), 6);
        assert_eq!(automaton.all_transitions(automaton.root()).count(), 1);
        assert_eq!(automaton.all_transitions(root_child(&automaton)).count(), 4);
    }
}
