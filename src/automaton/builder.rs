use std::hash::Hash;

use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph, Direction};

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    constraint::Constraint,
    mutex_tree::{MutuallyExclusiveTree, ToConstraintsTree},
    utils::{subgraph::SubgraphRef, OnlineToposort},
    HashMap, HashSet, PatternID,
};

mod acyclic;
mod modify;

use self::acyclic::AcyclicChecker;

use super::{State, Transition, TransitionID};

/// A predicate on a list of constraints
pub type ConstraintListPredicate<'c, C> = Box<dyn FnMut(&[&C]) -> bool + 'c>;

/// Create constraint automata from lists of patterns, given by lists of
/// constraints.
pub struct AutomatonBuilder<'c, C, I> {
    /// The matcher being built
    matcher: ConstraintAutomaton<C, I>,
    /// The list of all patterns IDs, in order of addition
    patterns_ids: Vec<PatternID>,
    /// The heuristic used to determine whether to turn a state into a deterministic
    /// one.
    make_det: Option<ConstraintListPredicate<'c, C>>,
    /// The list of all nodes that were added in the last iteration
    recently_added: HashSet<NodeIndex>,
}

impl<'c, C: Eq + Clone, I> AutomatonBuilder<'c, C, I> {
    /// Construct an empty automaton builder.
    pub fn new() -> Self
    where
        I: Default,
    {
        Self {
            matcher: ConstraintAutomaton::new(),
            patterns_ids: Vec::new(),
            make_det: None,
            recently_added: HashSet::default(),
        }
    }

    fn with_indexing_scheme(host_indexing: I) -> Self {
        Self {
            matcher: ConstraintAutomaton::with_indexing_scheme(host_indexing),
            patterns_ids: Vec::new(),
            make_det: None,
            recently_added: HashSet::default(),
        }
    }

    /// Set the deterministic heuristic to use.
    ///
    /// The heuristic is a function that takes a list of constraints and returns
    /// true if the state should be turned into a deterministic one.
    pub fn set_det_heuristic(self, make_det: impl FnMut(&[&C]) -> bool + 'c) -> Self {
        Self {
            matcher: self.matcher,
            patterns_ids: self.patterns_ids,
            make_det: Some(Box::new(make_det)),
            recently_added: self.recently_added,
        }
    }

    /// Construct an automaton builder from a list of patterns, given by lists of
    /// constraints.
    ///
    /// Use `I::default()` as the indexing scheme.
    pub fn from_constraints(patterns: impl IntoIterator<Item = Vec<C>>) -> Self
    where
        I: Default,
    {
        Self::from_constraints_with_index_scheme(patterns, I::default())
    }

    /// Construct an automaton builder from a list of patterns with a custom
    /// indexing scheme.
    pub fn from_constraints_with_index_scheme(
        patterns: impl IntoIterator<Item = Vec<C>>,
        host_indexing: I,
    ) -> Self {
        patterns.into_iter().fold(
            Self::with_indexing_scheme(host_indexing),
            |mut builder, pattern| {
                builder.add_pattern(pattern);
                builder
            },
        )
    }

    /// Add a pattern to the automaton builder.
    pub fn add_pattern(&mut self, pattern: Vec<C>) {
        let id = PatternID(self.patterns_ids.len());
        self.matcher.add_pattern(pattern, id);
        self.patterns_ids.push(id);
    }
}

type Automaton<K, P, I> = ConstraintAutomaton<Constraint<K, P>, I>;

impl<'c, K, P, I> AutomatonBuilder<'c, Constraint<K, P>, I>
where
    Constraint<K, P>: Eq + Clone + Hash + ToConstraintsTree,
{
    /// Construct the automaton.
    ///
    /// The returned automaton will be able to match `self.patterns` and will
    /// respect the automaton invariants:
    ///  - all outgoing transitions are unique at every state
    ///  - all outgoing transitions are mutually exclusive
    ///
    /// The `make_det` predicate specifies the heuristic used to determine whether
    /// to turn a state into a deterministic one. To reduce the automaton size,
    /// states are merged whenever possible.
    pub fn finish(mut self) -> (Automaton<K, P, I>, Vec<PatternID>) {
        // Traverse the prefix tree from root to leaves and make the invariants
        // hold. The changes only affect nodes in the future of the root, i.e.
        // nodes on which the invariant does not hold yet.

        // With this `toposort` we are allowed to add vertices and edges as we go
        let mut traverser = self.matcher.toposort();
        while let Some(state) = traverser.next(&self.matcher) {
            if self.matcher.state_exists(state) {
                // `state` was deleted, skip
                continue;
            }
            // Group all identical constraints and FAIL transitions into one
            self.make_constraints_unique(state);

            // Make the mutually exclusivity invariant hold
            self.make_mutex(state);

            // Turn some of the states into deterministic transitions, according to
            // `make_det`
            let constraints = self.matcher.constraints(state).collect_vec();
            if self.make_det.as_mut().unwrap()(&constraints) {
                self.make_det(state);
            }

            // For all nodes that were added try to merge them with existing
            // ones.
            self.try_merge_new_nodes();
        }
        (self.matcher, self.patterns_ids)
    }

    /// Turn outgoing constraints at `state` into a mutually exclusive set.
    ///
    /// Use `ToMutuallyExclusiveTree` to turn the constraints into a mutually
    /// exclusive tree, which is inserted in place of `state`.
    ///
    /// This may insert epsilon transitions, i.e. edges with no associated
    /// constraint. These will always be last in the constraint order.
    fn make_mutex(&mut self, state: StateID) {
        if self.matcher.is_deterministic(state) {
            // Nothing to do
            return;
        }

        if !self.matcher.constraints(state).any(|_| true) {
            // There are no constraints, already in a deterministic state
            return;
        }
        // Disconnect all non-fail children
        let constraints_children = self.matcher.drain_constraints(state).collect_vec();

        // Filter out None constraint
        let (constraints, children): (Vec<_>, Vec<_>) = constraints_children
            .into_iter()
            .filter_map(|(cons, child)| Some((cons?, child)))
            .unzip();

        // Organise constraints into a tree of mutually exclusive constraints
        let mutex_tree = Constraint::to_constraints_tree(constraints.clone());

        let added_constraints = self.add_mutex_tree(mutex_tree, state, &children);

        // All constraints that were not present in the mutex tree are added as
        // children of an epsilon transition at `state`.
        let not_added = (0..constraints.len())
            .filter(|i| !added_constraints.contains(i))
            .collect_vec();

        // Add/hide any constraints that were not added under a fail transition
        if !not_added.is_empty() {
            let fail_state = self.add_fail(state);
            // Add edges to children
            for i in not_added {
                self.matcher
                    .append_edge(fail_state, children[i], Some(constraints[i].clone()));
            }
        }
    }
}

impl<'c, C: Clone + Eq + Hash, I> AutomatonBuilder<'c, C, I> {
    fn make_constraints_unique(&mut self, state: StateID) {
        let mut grouped_transitions = HashMap::default();
        for t in self.matcher.all_transitions(state) {
            let cons = self.matcher.constraint(t).cloned();
            grouped_transitions
                .entry(cons)
                .or_insert_with(Vec::new)
                .push(t);
        }
        for transitions in grouped_transitions.into_values() {
            // merge all `transitions` into a single one.
            if transitions.len() <= 1 {
                // nothing to merge
                continue;
            }

            let old_children = transitions
                .iter()
                .map(|&t| self.matcher.next_state(t))
                .unique()
                .collect_vec();

            // Remove transitions
            let mut removed_transition = None;
            for &t in &transitions {
                removed_transition = Some(self.matcher.remove_transition(t));
            }
            let Transition { constraint, .. } = removed_transition.unwrap();

            // Add a single transition with the same constraint
            let new_child = self.add_transition(state, constraint);

            // Copy all transitions of `old_children` to `new_child`
            for &old_child in &old_children {
                self.matcher.clone_outgoing(new_child, old_child);
                let pattern_matches = self.matcher.matches(old_child).to_vec();
                for pattern in pattern_matches {
                    self.matcher.add_match(new_child, pattern);
                }
                if self.matcher.is_unreachable(old_child) {
                    self.matcher.remove_state(old_child);
                }
            }
        }
    }

    /// Make the given state deterministic
    ///
    /// Expects that all constraints are mutually exclusive and there is at
    /// most one fail transition.
    ///
    /// Achieved by adding all transitions of the fail state to all other children
    /// of `state`.
    fn make_det(&mut self, state: StateID) {
        if self.matcher.set_deterministic(state) {
            // Already deterministic
            return;
        }
        let Some(fail_state) = self.matcher.fail_next_state(state) else {
            return;
        };
        let transitions_fail = self.matcher.all_transitions(fail_state).collect_vec();
        for transition in self.matcher.all_constraint_transitions(state).collect_vec() {
            let target = self.matcher.split_target(transition);
            self.recently_added.insert(target.0);
            for &t in &transitions_fail {
                self.matcher.append_edge(
                    target,
                    self.matcher.next_state(t),
                    self.matcher.constraint(t).cloned(),
                );
            }
        }
    }

    fn recently_added_subgraph(
        &self,
    ) -> SubgraphRef<'_, StableDiGraph<State, Transition<C>>, NodeIndex> {
        SubgraphRef::new(&self.matcher.graph, &self.recently_added, true)
    }

    /// Attempt to merge all recently added nodes with its siblings.
    ///
    /// This is an optimisation that aims to reduce the number of states
    /// in the automaton. This leaves the behaviour of the state automaton
    /// unchanged.
    fn try_merge_new_nodes(&mut self) {
        // We try to merge nodes in a (reverse) toposort order, as merging
        // children first might enable the merge of parent nodes.
        // We thus start at the newly added nodes that have no newly added descendants
        let roots = self.recently_added.iter().copied().filter(|&n| {
            self.matcher
                .graph
                .neighbors_directed(n, Direction::Outgoing)
                .filter(|n| self.recently_added.contains(n))
                .count()
                == 0
        });
        let mut traverser = OnlineToposort::from_iter(roots);
        let mut acyclic_checker = AcyclicChecker::with_graph(&self.matcher.graph).unwrap();

        while let Some(node) = traverser.next(self.recently_added_subgraph()) {
            // Try to merge node with its siblings
            let state = StateID(node);
            let Some(first_child) = self
                .matcher
                .all_transitions(state)
                .next()
                .map(|t| self.matcher.next_state(t))
            else {
                continue;
            };
            let siblings = self
                .matcher
                .incoming_transitions(first_child)
                .map(|t| self.matcher.parent(t))
                .unique()
                .filter(|&n| n != state)
                // We cannot merge nodes that are reachable from each other
                .filter(|n| acyclic_checker.path_exists(n.0, state.0, &self.matcher.graph));
            let state_tuple = self.matcher.state_tuple(state);
            let merge_nodes = siblings
                .filter(|&n| self.matcher.state_tuple(n) == state_tuple)
                .collect_vec();
            // merge all `nodes` into a single one.
            if merge_nodes.len() <= 1 {
                // nothing to merge
                continue;
            }
            let first_node = merge_nodes[0];
            for &node in &merge_nodes[1..] {
                self.matcher.move_incoming(first_node, node);
                self.matcher.remove_state(node);
            }
            acyclic_checker.merge_nodes(merge_nodes.into_iter().map(|n| n.0));
        }
        self.recently_added.clear();
    }

    fn add_mutex_tree(
        &mut self,
        mutex_tree: MutuallyExclusiveTree<C>,
        state: StateID,
        children: &[StateID],
    ) -> HashSet<usize> {
        let mut added_constraints = HashSet::default();

        // Traverse the mutex tree, making sure to
        //  - add each state of the tree to the matcher as we go
        //  - keep track of the matcher states corresponding to tree states
        //  - add edges to children when the constraint index is set
        let mut curr_states = vec![(mutex_tree.root(), state)];
        while let Some((tree_state, matcher_state)) = curr_states.pop() {
            for (child_tree_state, c) in mutex_tree.children(tree_state) {
                if mutex_tree.n_children(child_tree_state) > 0 {
                    let child_matcher_state = self.add_constraint(matcher_state, c.clone());
                    curr_states.push((child_tree_state, child_matcher_state));
                }

                let indices = mutex_tree.constraint_indices(child_tree_state);
                for &index in indices {
                    added_constraints.insert(index);
                }
                for &ind in indices {
                    self.matcher
                        .append_edge(matcher_state, children[ind], Some(c.clone()));
                }
            }
        }
        added_constraints
    }

    fn add_fail(&mut self, state: StateID) -> StateID {
        self.add_transition(state, None)
    }

    fn add_transition(&mut self, state: StateID, constraint: Option<C>) -> StateID {
        let new_state = self.matcher.add_transition(state, constraint);
        self.recently_added.insert(new_state.0);
        new_state
    }

    fn add_constraint(&mut self, state: StateID, constraint: C) -> StateID {
        self.add_transition(state, Some(constraint))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracedStateID {
    state: StateID,
    node: NodeIndex,
}

impl From<TracedStateID> for StateID {
    fn from(state: TracedStateID) -> Self {
        state.state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracedTransitionID {
    transition: TransitionID,
    parent: NodeIndex,
}

impl From<TracedTransitionID> for TransitionID {
    fn from(transition: TracedTransitionID) -> Self {
        transition.transition
    }
}

impl<'c, C: Eq + Clone, I: Default> Default for AutomatonBuilder<'c, C, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'c, C: Clone + Eq, I: Default> FromIterator<Vec<C>> for AutomatonBuilder<'c, C, I> {
    fn from_iter<T: IntoIterator<Item = Vec<C>>>(iter: T) -> Self {
        Self::from_constraints(iter)
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_snapshot;
    use rstest::rstest;
    use tests::modify::tests::{constraints, root_child, root_grandchildren};

    use super::modify::tests::{automaton, automaton2};
    use crate::{
        constraint::tests::TestConstraint, indexing::tests::TestIndexingScheme,
        mutex_tree::MutuallyExclusiveTree,
    };

    use super::*;

    // Dummy ToMutuallyExclusiveTree implementation for TestConstraint
    impl ToConstraintsTree for TestConstraint {
        fn to_constraints_tree(preds: Vec<Self>) -> MutuallyExclusiveTree<Self> {
            // We take the first `k` constraints to be mutually exclusive,
            // where `k` is given by the arity of the first predicate (this has
            // no meaning).
            let (inds, preds): (Vec<_>, Vec<_>) = preds
                .into_iter()
                .enumerate()
                .sorted_by(|(_, p1), (_, p2)| p1.cmp(p2))
                .unzip();
            let Some(k) = preds.first().map(|c| c.predicate().arity) else {
                return MutuallyExclusiveTree::new();
            };
            let first_k = preds.into_iter().take(k);
            let mut tree = MutuallyExclusiveTree::new();
            let new_children = tree.add_children(tree.root(), first_k).collect_vec();
            for (index, child) in inds.into_iter().zip(new_children) {
                tree.add_constraint_index(child, index).unwrap();
            }
            tree
        }
    }

    impl<C, I> ConstraintAutomaton<C, I> {
        fn wrap_builder(self, with_builder: impl Fn(&mut AutomatonBuilder<'static, C, I>)) -> Self {
            let mut builder = AutomatonBuilder {
                matcher: self,
                patterns_ids: Vec::new(),
                make_det: None,
                recently_added: HashSet::default(),
            };
            with_builder(&mut builder);
            builder.matcher
        }
    }

    #[test]
    fn test_add_mutex_tree() {
        let mut builder = AutomatonBuilder::<_, TestIndexingScheme>::default();
        let n2 = builder.matcher.add_non_det_node();
        let mutex_tree = {
            let mut tree = MutuallyExclusiveTree::new();
            let tree_child = tree.add_child(tree.root(), ());
            tree.add_constraint_index(tree_child, 0).unwrap();
            let tree_gchild = tree.add_child(tree_child, ());
            tree.add_constraint_index(tree_gchild, 1).unwrap();
            tree
        };
        builder.add_mutex_tree(mutex_tree, builder.matcher.root(), &[n2, n2]);
        assert_snapshot!(builder.matcher.dot_string());
    }

    #[test]
    fn test_build() {
        let mut builder = AutomatonBuilder::<TestConstraint, TestIndexingScheme>::new()
            .set_det_heuristic(|_| false);
        let [constraint_root, _, constraint_b, constraint_c, constraint_d] = constraints();
        builder.add_pattern(vec![constraint_root.clone(), constraint_b.clone()]);
        builder.add_pattern(vec![constraint_root.clone(), constraint_c.clone()]);
        builder.add_pattern(vec![constraint_root, constraint_d.clone()]);
        let (matcher, pattern_ids) = builder.finish();
        assert_eq!(matcher.graph.node_count(), 6);
        assert_eq!(pattern_ids, vec![PatternID(0), PatternID(1), PatternID(2)]);
        let x_child = matcher.children(matcher.root()).exactly_one().ok().unwrap();

        // The two first patterns were kept at the root
        assert_eq!(
            matcher
                .all_transitions(x_child)
                .map(|t| { matcher.constraint(t) })
                .collect::<HashSet<_>>(),
            HashSet::from_iter([Some(&constraint_b), Some(&constraint_c), None])
        );
        // The remaining two patterns are children of an epsilon transition
        let fail_state = matcher.fail_next_state(x_child).unwrap();
        assert_eq!(
            matcher
                .all_transitions(fail_state)
                .map(|t| { matcher.constraint(t) })
                .collect::<HashSet<_>>(),
            HashSet::from_iter([Some(&constraint_d)])
        );
    }

    #[rstest]
    fn test_make_det_noop(automaton: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        let automaton2 = automaton.clone();
        let root_child = root_child(&automaton);
        let automaton = automaton.wrap_builder(|b| b.make_det(root_child));

        assert_eq!(automaton.graph.node_count(), automaton2.graph.node_count());
        assert_eq!(automaton.graph.edge_count(), automaton2.graph.edge_count());
    }

    #[rstest]
    fn test_make_det(automaton2: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        let x_child = root_child(&automaton2);
        let [a_child, b_child, c_child, d_child] = root_grandchildren(&automaton2);

        // Add a FAIL transition from x_child to a new state
        let fail_child = automaton2.fail_next_state(x_child).unwrap();
        // Add a common constraint to the fail child and a_child
        let common_constraint = TestConstraint::new(vec![7, 8]);
        let post_fail = automaton2
            .constraint_next_state(fail_child, &common_constraint)
            .unwrap();

        let automaton2 = automaton2.wrap_builder(|b| b.make_det(x_child));

        // Now `common_constraint` should be on all children, pointing to
        // `post_fail`. For b, c, d, this should be the only transition.
        for child in [b_child, c_child, d_child] {
            assert_eq!(automaton2.all_transitions(child).count(), 1);
            let transition = automaton2.all_transitions(child).next().unwrap();
            assert_eq!(automaton2.constraint(transition), Some(&common_constraint));
            assert_eq!(automaton2.next_state(transition), post_fail);
        }
        {
            // For a_child, there are two transitions with the same constraint
            assert_eq!(automaton2.all_transitions(a_child).count(), 2);
            let (t1, t2) = automaton2.all_transitions(a_child).collect_tuple().unwrap();
            assert_eq!(automaton2.constraint(t1), Some(&common_constraint));
            assert_eq!(automaton2.constraint(t2), Some(&common_constraint));
        }
    }

    #[rstest]
    fn test_make_unique(automaton2: ConstraintAutomaton<TestConstraint, TestIndexingScheme>) {
        let x_child = root_child(&automaton2);
        let [a_child, ..] = root_grandchildren(&automaton2);

        // FAIL transition from x_child to a new state
        let fail_child = automaton2.fail_next_state(x_child).unwrap();
        // Common constraint to both fail_child and a_child
        let common_constraint = TestConstraint::new(vec![7, 8]);

        let post_fail = automaton2
            .constraint_next_state(fail_child, &common_constraint)
            .unwrap();

        let automaton2 = automaton2.wrap_builder(|b| b.make_det(x_child));
        // Now a_child has two (identical) constraints
        assert_eq!(automaton2.all_transitions(a_child).count(), 2);

        let automaton2 = automaton2.wrap_builder(|b| b.make_constraints_unique(a_child));
        // Now child_a should have only one constraint
        assert_eq!(automaton2.all_transitions(a_child).count(), 1);

        let post_a = automaton2
            .constraint_next_state(a_child, &common_constraint)
            .unwrap();
        // And its child should have only one incoming transition (from a_child)
        assert_eq!(automaton2.incoming_transitions(post_a).count(), 1);

        // Meanwhile the child of fail has the other four states
        // (b_child, c_child, d_child and fail) as incoming transitions
        assert_eq!(automaton2.incoming_transitions(post_fail).count(), 4);

        // The child of a_child and of the other states merge again one state later
        let common_constraint2 = TestConstraint::new(vec![77, 8]);
        let post_post_fail = automaton2
            .constraint_next_state(post_fail, &common_constraint2)
            .unwrap();
        assert_eq!(
            automaton2
                .constraint_next_state(post_a, &common_constraint2)
                .unwrap(),
            post_post_fail
        );
        assert_eq!(
            automaton2
                .constraint_next_state(post_fail, &common_constraint2)
                .unwrap(),
            post_post_fail
        );
        assert_eq!(automaton2.incoming_transitions(post_post_fail).count(), 2);
    }
}
