use itertools::Itertools;

use crate::automaton::{ConstraintAutomaton, State, StateID, Transition, TransitionID};
use crate::indexing::IndexKey;
use crate::{Constraint, HashSet, BindMap, IndexingScheme, PatternID};

/// Methods for modifying the automaton
impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I>
where
    Constraint<K, P>: Eq + Clone,
{
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

    /// Add a transition on the transition graph from `parent` to `child` with
    /// the given `constraint`.
    ///
    /// The edge is appended at the end of the edge list, with the caveat that
    /// the edge order invariant that `None` is always at the end must be maintained.
    /// Thus if a fail transition is present and `constraint` is not None, the
    /// new transition is added at the penultimate position.
    ///
    /// The `constraint` must not exist at `parent`.
    pub(super) fn append_edge(
        &mut self,
        StateID(parent): StateID,
        StateID(child): StateID,
        constraint: Option<Constraint<K, P>>,
    ) {
        if parent == child {
            // Transitions to itself are useless
            return;
        }

        let is_fail_transition = constraint.is_none();
        let new_edge = self
            .graph
            .add_edge(parent, child, Transition { constraint });
        if is_fail_transition {
            self.node_weight_mut(StateID(parent))
                .epsilon_order
                .push(TransitionID(new_edge));
        } else {
            self.node_weight_mut(StateID(parent))
                .constraint_order
                .push(TransitionID(new_edge));
        }
    }

    /// Add a transition from `parent` to a new node with the given `constraint`.
    pub(in crate::automaton) fn add_constraint(
        &mut self,
        parent: StateID,
        constraint: Constraint<K, P>,
    ) -> StateID {
        self.add_transition(parent, Some(constraint))
    }

    pub(super) fn add_transition(
        &mut self,
        parent: StateID,
        constraint: Option<Constraint<K, P>>,
    ) -> StateID {
        let child = self.add_non_det_node();
        self.append_edge(parent, child, constraint);
        child
    }

    /// Add an epsilon transition from `parent`.
    #[cfg(test)]
    pub(in crate::automaton) fn add_fail(&mut self, parent: StateID) -> StateID {
        self.add_transition(parent, None)
    }

    pub(super) fn add_match(&mut self, state: StateID, pattern: PatternID, scope: HashSet<K>) {
        if !self.node_weight(state).matches.contains_key(&pattern) {
            self.node_weight_mut(state).matches.insert(pattern, scope);
        }
    }

    /// Add a pattern to the automaton.
    ///
    /// Add a pattern to the automaton with the given `id`. The pattern is
    /// represented by a list of `constraints`.
    ///
    /// The `required_bindings` define the domain of the pattern, i.e., the set
    /// of index keys that will be mapped to values in the data by a pattern
    /// match. The required bindings will always include all keys that appear
    /// in the pattern's constraints, but the argument can be used to specify
    /// additional required bindings. Pass `None` otherwise.
    pub(crate) fn add_pattern<M>(
        &mut self,
        constraints: impl IntoIterator<Item = Constraint<K, P>>,
        id: PatternID,
        required_bindings: impl IntoIterator<Item = K>,
    ) where
        // A complicated way of saying Key<I> == K
        I: IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
    {
        let mut state = self.root();
        let mut required_bindings: HashSet<_> = required_bindings.into_iter().collect();
        for constraint in constraints {
            required_bindings.extend(self.host_indexing.all_missing_bindings(
                constraint.required_bindings().iter().copied(),
                required_bindings.iter().copied(),
            ));
            state = self.add_constraint(state, constraint);
        }
        self.add_match(state, id, required_bindings);
    }

    /// Split the `transition`'s target and return the new state.
    ///
    /// The new state will have `transition` as its unique incoming edge. All
    /// other incoming transitions will remain in the existing target.
    pub(super) fn split_target(&mut self, transition: TransitionID) -> StateID
    where
        K: Clone,
    {
        let state = self.next_state(transition);
        if !self.incoming_transitions(state).any(|t| transition != t) {
            // There is a single incoming transition, nothing to do.
            return state;
        }
        let State {
            matches,
            deterministic,
            ..
        } = self.node_weight(state).clone();
        let new_state = {
            let node = self.graph.add_node(State {
                matches,
                deterministic,
                ..Default::default()
            });
            StateID(node)
        };
        self.rewire_target(transition, new_state);
        for t in self.all_transitions(state).collect_vec() {
            let target = self.next_state(t);
            self.append_edge(new_state, target, self.constraint(t).cloned());
        }
        new_state
    }

    fn rewire_target(&mut self, transition: TransitionID, new_target: StateID) -> TransitionID {
        let source = self.parent(transition);
        let weight = self.graph.remove_edge(transition.0).unwrap();
        let new_transition = TransitionID(self.graph.add_edge(source.0, new_target.0, weight));
        self.node_weight_mut(source)
            .replace_order(transition, new_transition);
        new_transition
    }

    pub(super) fn drain_constraints(
        &mut self,
        state: StateID,
    ) -> impl Iterator<Item = (Option<Constraint<K, P>>, StateID)> + '_ {
        let transitions = self.all_constraint_transitions(state).collect_vec();
        transitions.into_iter().map(|transition| {
            let target = self.next_state(transition);
            let constraint = self.remove_transition(transition).constraint;
            (constraint, target)
        })
    }

    pub(super) fn remove_transition(
        &mut self,
        transition: TransitionID,
    ) -> Transition<Constraint<K, P>> {
        self.remove_order(transition);
        self.graph
            .remove_edge(transition.0)
            .expect("invalid transition")
    }

    pub(super) fn remove_state(&mut self, state: StateID) {
        self.graph.remove_node(state.0);
    }

    /// Clone all outgoing transitions from `other_state` to `state`
    pub(super) fn clone_outgoing(&mut self, state: StateID, other_state: StateID) {
        for t in self.all_transitions(other_state).collect_vec() {
            let target = self.next_state(t);
            self.append_edge(state, target, self.constraint(t).cloned());
        }
    }

    /// Move all incoming transitions from `other_state` to `state`.
    ///
    /// As a result, `other_state` will have no incoming transitions and should
    /// be removed.
    pub(super) fn move_incoming(&mut self, state: StateID, other_state: StateID) {
        for t in self.incoming_transitions(other_state).collect_vec() {
            let source = self.parent(t);
            let constraint = self.remove_transition(t).constraint;
            self.append_edge(source, state, constraint);
        }
    }
}

// Small, private utils functions
impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I>
where
    Constraint<K, P>: Eq + Clone,
{
    fn remove_order(&mut self, transition: TransitionID) {
        let source = self.parent(transition);
        let order = if self.constraint(transition).is_none() {
            &mut self.node_weight_mut(source).epsilon_order
        } else {
            &mut self.node_weight_mut(source).constraint_order
        };
        let ind = order
            .iter()
            .position(|&e| e == transition)
            .expect("invalid transition");
        order.remove(ind);
    }

    fn node_weight_mut(&mut self, StateID(state): StateID) -> &mut State<K> {
        self.graph.node_weight_mut(state).expect("invalid state")
    }
}

impl<K: IndexKey> State<K> {
    fn replace_order(&mut self, old_transition: TransitionID, new_transition: TransitionID) {
        let State {
            constraint_order,
            epsilon_order,
            ..
        } = self;
        let mut all_order_mut = constraint_order.iter_mut().chain(epsilon_order.iter_mut());
        let order_mut = all_order_mut.find(|t| **t == old_transition).unwrap();
        *order_mut = new_transition;
    }
}

#[cfg(test)]
pub mod tests {
    use rstest::{fixture, rstest};

    use crate::{
        automaton::{tests::TestAutomaton, AutomatonBuilder},
        constraint::tests::TestConstraint,
        HashSet,
    };

    use super::*;

    /// An automaton with a X transition at the root and transitions
    /// [a,b,c,d] at the only child
    #[fixture]
    pub fn automaton() -> TestAutomaton {
        let mut builder = AutomatonBuilder::new().set_det_heuristic(|_| false);
        let [constraint_root, constraint_a, constraint_b, constraint_c, constraint_d] =
            constraints();
        builder.add_pattern(vec![constraint_root.clone(), constraint_a], 0, None);
        builder.add_pattern(vec![constraint_root.clone(), constraint_b], 1, None);
        builder.add_pattern(vec![constraint_root.clone(), constraint_c], 2, None);
        builder.add_pattern(vec![constraint_root, constraint_d], 3, None);
        builder.finish().0
    }

    #[fixture]
    pub fn automaton2() -> TestAutomaton {
        let mut automaton = automaton();
        let x_child = root_child(&automaton);
        let [a_child, _, _, _] = root_grandchildren(&automaton);

        // Add a FAIL transition from x_child to a new state
        let fail_child = automaton.add_fail(x_child);
        // Add a common constraint to the fail child and a_child
        let common_constraint = TestConstraint::new(vec![7, 8]);
        let post_fail = automaton.add_constraint(fail_child, common_constraint.clone());
        automaton.add_constraint(a_child, common_constraint.clone());
        // Add a second common constraint to post_fail
        let common_constraint2 = TestConstraint::new(vec![77, 8]);
        automaton.add_constraint(post_fail, common_constraint2.clone());
        automaton
    }

    pub fn constraints() -> [TestConstraint; 5] {
        [
            TestConstraint::new(vec![0]),
            TestConstraint::new(vec![0, 1, 2, 4]),
            TestConstraint::new(vec![1, 4]),
            TestConstraint::new(vec![2, 4]),
            TestConstraint::new(vec![3, 4]),
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

    #[rstest]
    fn test_drain_constraints(mut automaton: TestAutomaton) {
        let [_, constraint_a, constraint_b, constraint_c, constraint_d] = constraints();
        let [child_a, child_b, child_c, child_d] = root_grandchildren(&automaton);
        let drained: HashSet<_> = automaton
            .drain_constraints(root_child(&automaton))
            .collect();
        assert_eq!(
            drained,
            HashSet::from_iter([
                (Some(constraint_a), child_a),
                (Some(constraint_b), child_b),
                (Some(constraint_c), child_c),
                (Some(constraint_d), child_d)
            ])
        );
    }
}
