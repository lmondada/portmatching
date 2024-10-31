use std::{cmp::Reverse, hash::Hash};

use itertools::Itertools;
use petgraph::{
    acyclic::Acyclic,
    data::{DataMap, DataMapMut},
    graph::NodeIndex,
    visit::{
        Data, EdgeRef, GraphBase, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeIdentifiers,
        Reversed, Visitable,
    },
    Direction,
};

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    constraint::Constraint,
    constraint_tree::{ConstraintTree, ToConstraintsTree},
    indexing::IndexKey,
    BindMap, DetHeuristic, HashMap, HashSet, IndexingScheme, PatternID,
};

mod modify;

use super::{view::GraphView, State, Transition, TransitionGraph, TransitionID};

/// Create constraint automata from lists of patterns, given by lists of
/// constraints.
pub struct AutomatonBuilder<K: IndexKey, P, I> {
    /// The matcher being built, made of a graph, a root state and an indexing scheme
    graph: Acyclic<TransitionGraph<K, P>>,
    /// The root of the graph
    root: StateID,
    /// The indexing scheme used
    host_indexing: I,
    /// The list of all patterns IDs, in order of addition
    patterns_ids: Vec<PatternID>,
    /// The list of all nodes that were added in the last iteration
    recently_added: HashSet<StateID>,
}

impl<K: IndexKey, P, I> AutomatonBuilder<K, P, I> {
    /// Construct an empty automaton builder.
    pub fn new() -> Self
    where
        I: Default,
        P: 'static,
    {
        Self::with_indexing_scheme(I::default())
    }

    fn with_indexing_scheme(host_indexing: I) -> Self
    where
        P: 'static,
    {
        Self::from_matcher(ConstraintAutomaton::with_indexing_scheme(host_indexing))
    }

    fn from_matcher(automaton: ConstraintAutomaton<K, P, I>) -> Self
    where
        P: 'static,
    {
        let ConstraintAutomaton {
            graph,
            root,
            host_indexing,
        } = automaton;
        Self {
            graph: Acyclic::try_from_graph(graph).unwrap(),
            root,
            host_indexing,
            patterns_ids: Vec::new(),
            recently_added: HashSet::default(),
        }
    }
}

impl<K: IndexKey, P: 'static, I> AutomatonBuilder<K, P, I>
where
    Constraint<K, P>: Eq + Clone,
{
    /// Construct an automaton builder from a list of patterns, given by lists of
    /// constraints.
    ///
    /// Use `I::default()` as the indexing scheme.
    pub fn from_constraints<M>(patterns: impl IntoIterator<Item = Vec<Constraint<K, P>>>) -> Self
    where
        I: Default + IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
        K: 'static,
        P: 'static,
    {
        Self::from_constraints_with_index_scheme(patterns, I::default())
    }

    /// Construct an automaton builder from a list of patterns with a custom
    /// indexing scheme.
    pub fn from_constraints_with_index_scheme<M>(
        patterns: impl IntoIterator<Item = Vec<Constraint<K, P>>>,
        host_indexing: I,
    ) -> Self
    where
        I: IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
        K: 'static,
        P: 'static,
    {
        let mut pattern_id = 0;
        patterns.into_iter().fold(
            Self::with_indexing_scheme(host_indexing),
            |mut builder, pattern| {
                builder.add_pattern(pattern, PatternID(pattern_id), None);
                pattern_id += 1;
                builder
            },
        )
    }

    fn find_mergeable_states(&self, state: StateID) -> Vec<StateID> {
        let siblings = {
            let Some(first_child) = self
                .all_transitions(state)
                .next()
                .map(|t| self.next_state(t))
            else {
                return vec![];
            };
            self.incoming_transitions(first_child)
                .map(|t| self.parent(t))
        };

        let mut no_path_siblings = HashSet::from_iter([state]);
        for sibling in siblings.filter(|&n| n != state).unique() {
            if !path_exists(sibling, no_path_siblings.iter().copied(), &self.graph) {
                no_path_siblings.insert(sibling);
            }
        }

        let state_tuple = self.state_tuple(state);
        no_path_siblings
            .into_iter()
            .map_into()
            .filter(|&n| self.state_tuple(n) == state_tuple)
            .collect_vec()
    }

    /// Get the root of the automaton.
    pub(super) fn root(&self) -> StateID {
        self.root
    }
}

/// Check if there is a path from/to `node` to/from any of the nodes in
/// `other_nodes`.
fn path_exists<K: IndexKey, P: 'static>(
    node: StateID,
    other_nodes: impl IntoIterator<Item = StateID>,
    graph: &Acyclic<TransitionGraph<K, P>>,
) -> bool {
    other_nodes
        .into_iter()
        .any(|n| !graph.is_valid_edge(node.0, n.0) || !graph.is_valid_edge(n.0, node.0))
}

impl<K: IndexKey, P, I> AutomatonBuilder<K, P, I>
where
    P: ToConstraintsTree<K> + 'static,
    Constraint<K, P>: Eq + Clone + Hash,
    K: Clone,
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
    pub fn finish<M>(self) -> (ConstraintAutomaton<K, P, I>, Vec<PatternID>)
    where
        I: IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
    {
        self.finish_with_det_heuristic(DetHeuristic::Default)
    }

    pub(crate) fn finish_with_det_heuristic<M>(
        mut self,
        det_heuristic: DetHeuristic<K, P>,
    ) -> (ConstraintAutomaton<K, P, I>, Vec<PatternID>)
    where
        I: IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
    {
        // Traverse the prefix tree from root to leaves and make the invariants
        // hold. The changes only affect nodes in the future of the root, i.e.
        // nodes on which the invariant does not hold yet.

        // We traverse the graph in toposort order, with the twist that we are
        // adding nodes and vertices as we go. We ensure this is well defined by
        // only adding nodes in the future of the current node. Hence, we can
        // keep a current `pos` in the topological ordering: everything before
        // `pos` is immutable, whereas everything after `pos` is mutable.
        let mut pos = 0;
        while pos < self.graph.node_count() {
            let state = StateID(self.graph.order()[pos]);

            // Group all identical constraints and FAIL transitions into one
            self.make_constraints_unique(state);

            // Make the mutually exclusivity invariant hold
            let make_det = self.insert_constraint_tree(state);

            // The invariant might have been broken by the previous step
            self.make_constraints_unique(state);

            // Turn some of the states into deterministic transitions, according to
            // `make_det`
            if make_det {
                let constraints = self.constraints(state).collect_vec();
                if det_heuristic.make_det(&constraints) {
                    self.make_det(state);
                    // Add `state` to the set of recently added nodes as it has been changed
                    self.recently_added.insert(state);
                }
            }

            // For all nodes that were added try to merge them with existing
            // ones.
            self.try_merge_new_nodes();

            // The pos of state might have changed a bit. If this moves backwarks,
            // then we will have to retraverse a couple of nodes -- not an issue
            // in practice. If this is moving forwards, then just process the new
            // node at `pos` again (this should only happen when an edge
            // a -> state is added).
            let new_pos = self.graph.get_position(state.0);
            if new_pos <= pos {
                pos = new_pos + 1;
            }
        }

        // Now traverse from end to front to save the scope of each automaton state.
        self.populate_scopes();

        let matcher = ConstraintAutomaton {
            graph: self.graph.into_inner(),
            root: self.root,
            host_indexing: self.host_indexing,
        };
        (matcher, self.patterns_ids)
    }

    /// Populate the scope of each automaton state.
    ///
    /// The scope is the set of bindings that are "relevant" to the current
    /// traversal. They are the set of bindings that
    ///  - are in all paths from root to `state`, and
    ///  - appear in at least one pattern match in the future of `state`, or
    ///  - are required to evaluate outgoing constraints at `state`.
    fn populate_scopes<M>(&mut self)
    where
        I: IndexingScheme<BindMap = M>,
        M: BindMap<Key = K>,
    {
        // The child scope is obtained by:
        // - taking the intersection of the parents' scopes
        // - adding the constraints of the current edge
        let intersect_vec = |mut a: Vec<_>, b: Vec<_>| {
            a.retain(|x| b.contains(x));
            a
        };
        let mut forward_scopes =
            compute_scopes(self.graph.inner(), intersect_vec, |edge, known_bindings| {
                let new_bindings = edge
                    .weight()
                    .constraint
                    .as_ref()
                    .into_iter()
                    .flat_map(|c| c.required_bindings())
                    .copied();
                self.host_indexing
                    .all_missing_bindings(new_bindings, known_bindings.iter().copied())
            });

        // The parent scope is obtained by:
        // - taking the union of the children's scopes
        // - adding the required bindings of the matches at the child
        let union_set = |mut a: HashSet<_>, b| {
            a.extend(b);
            a
        };
        let mut backward_scopes =
            compute_scopes(Reversed(self.graph.inner()), union_set, |edge, _| {
                let s = StateID(edge.source());
                let pattern_matches = self.matches(s);
                pattern_matches
                    .values()
                    .flat_map(|s| s.iter())
                    .copied()
                    .collect()
            });

        // Intersect the two scopes, and add the bindings required to evaluate
        // the outgoing constraints at `state`
        let all_nodes = self.graph.node_indices().collect_vec();
        for node in all_nodes {
            let mut forward_scope = forward_scopes.remove(&node).unwrap();
            let backward_scope = backward_scopes.remove(&node).unwrap();
            forward_scope.retain(|k| backward_scope.contains(k));
            let mut scope = forward_scope;

            scope.extend(
                self.host_indexing.all_missing_bindings(
                    self.constraints(StateID(node))
                        .flat_map(|c| c.required_bindings())
                        .copied(),
                    scope.iter().copied(),
                ),
            );
            self.graph.node_weight_mut(node).unwrap().required_bindings = scope;
        }
    }

    /// Decompose outgoing constraints at `state` by replacing them with a
    /// constraint tree.
    ///
    /// Return whether the root state should be made deterministic.
    ///
    /// Use [`ToConstraintsTree`] to turn the constraints into a constraint
    /// tree, which is inserted in place of `state`.
    ///
    /// This may insert epsilon transitions, i.e. edges with no associated
    /// constraint. These will always be last in the constraint order.
    fn insert_constraint_tree(&mut self, state: StateID) -> bool {
        if self.is_deterministic(state) {
            // Nothing to do
            return false;
        }

        if !self.constraints(state).any(|_| true) {
            // There are no constraints, already in a deterministic state
            return false;
        }
        // Disconnect all non-fail children
        let constraints_children = self.drain_constraints(state).collect_vec();

        // Filter out None constraint
        let (constraints, children): (Vec<_>, Vec<_>) = constraints_children
            .into_iter()
            .filter_map(|(cons, child)| Some((cons?, child)))
            .unzip();

        // Organise constraints into a tree of mutually exclusive constraints
        let constraint_tree = P::to_constraints_tree(constraints.clone());
        let make_det = constraint_tree.make_det;

        let added_constraints = self.add_constraint_tree(constraint_tree, state, &children);

        // All constraints that were not present in the constraint tree are added as
        // children of an epsilon transition at `state`.
        let not_added = (0..constraints.len())
            .filter(|i| !added_constraints.contains(i))
            .collect_vec();

        // Add any constraints that were not added under a fail transition
        if !not_added.is_empty() {
            let fail_state = self.add_fail(state);
            // Add edges to children
            for i in not_added {
                self.append_edge(fail_state, children[i], Some(constraints[i].clone()));
            }
        }

        make_det
    }

    fn make_constraints_unique(&mut self, state: StateID) {
        let mut grouped_transitions = HashMap::default();
        for t in self.all_transitions(state) {
            let cons = self.constraint(t).cloned();
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
                .map(|&t| self.next_state(t))
                .unique()
                .collect_vec();

            // Remove transitions
            let mut removed_transition = None;
            for &t in &transitions {
                removed_transition = Some(self.remove_transition(t));
            }
            let Transition { constraint, .. } = removed_transition.unwrap();

            // Add a single transition with the same constraint
            let new_child = self.add_transition(state, constraint);

            // Copy all transitions of `old_children` to `new_child`
            for &old_child in &old_children {
                self.clone_outgoing(new_child, old_child);
                let pattern_matches = self.matches(old_child).clone();
                for (pattern, bindings) in pattern_matches {
                    self.add_match(new_child, pattern, bindings);
                }
                if self.is_unreachable(old_child) {
                    self.remove_state(old_child);
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
        if self.set_deterministic(state) {
            // Already deterministic
            return;
        }
        let Some(fail_state) = self.fail_next_state(state) else {
            return;
        };
        let transitions_fail = self.all_transitions(fail_state).collect_vec();
        for transition in self.all_constraint_transitions(state).collect_vec() {
            let target = self.split_target(transition);
            self.recently_added.insert(target);
            for &t in &transitions_fail {
                self.append_edge(target, self.next_state(t), self.constraint(t).cloned());
            }
        }
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
        let recently_added = self
            .recently_added
            .drain()
            .sorted_by_key(|&n| Reverse(self.graph.get_position(n.0)))
            .collect_vec();

        let mut to_remove = HashSet::default();

        for state in recently_added {
            if to_remove.contains(&state) {
                continue;
            }
            // Find all siblings that can be merged with `node`
            let merge_states = self.find_mergeable_states(state);

            // merge all `nodes` into a single one.
            if merge_states.len() <= 1 {
                // nothing to merge
                continue;
            }
            let first_state = merge_states[0];
            for &state in &merge_states[1..] {
                self.move_incoming(first_state, state);
                to_remove.insert(state);
            }
        }
        for state in to_remove {
            self.remove_state(state);
        }
    }

    fn add_constraint_tree(
        &mut self,
        constraint_tree: ConstraintTree<Constraint<K, P>>,
        state: StateID,
        children: &[StateID],
    ) -> HashSet<usize> {
        let mut added_constraints = HashSet::default();

        // Add FAIL transitions for constraints at the root of the constraint tree
        for &ind in constraint_tree.constraint_indices(constraint_tree.root()) {
            added_constraints.insert(ind);
            self.append_edge(state, children[ind], None);
        }

        // Traverse the constraint tree, making sure to
        //  - add each state of the tree to the matcher as we go
        //  - keep track of the matcher states corresponding to tree states
        //  - add edges to children when the constraint index is set
        let mut curr_states = vec![(constraint_tree.root(), state)];
        while let Some((tree_state, matcher_state)) = curr_states.pop() {
            for (child_tree_state, c) in constraint_tree.children(tree_state) {
                if constraint_tree.n_children(child_tree_state) > 0 {
                    let child_matcher_state = self.add_constraint(matcher_state, c.clone());
                    curr_states.push((child_tree_state, child_matcher_state));
                }

                // Add edges to the final children for `indices`
                let indices = constraint_tree.constraint_indices(child_tree_state);
                for &ind in indices {
                    added_constraints.insert(ind);
                    self.append_edge(matcher_state, children[ind], Some(c.clone()));
                }
            }
        }
        added_constraints
    }
}

fn compute_scopes<K, P, G, Container>(
    graph: G,
    merge_scopes: impl Fn(Container, Container) -> Container,
    get_scope: impl Fn(G::EdgeRef, &Container) -> Container,
) -> HashMap<NodeIndex, Container>
where
    K: IndexKey,
    G: Data<EdgeWeight = Transition<Constraint<K, P>>, NodeWeight = State<K>>,
    G: IntoNeighborsDirected + IntoNodeIdentifiers + Visitable + IntoEdgesDirected,
    G: GraphBase<NodeId = NodeIndex>,
    G: DataMap,
    Container: Extend<K> + Default + Clone + IntoIterator<Item = K>,
{
    let topo_order = petgraph::algo::toposort(&graph, None).expect("Graph should be acyclic");

    let mut ret: HashMap<NodeIndex, Container> = HashMap::default();

    // Iterate over the nodes in topological order
    for node_index in topo_order {
        // Get all parents in the graph
        let parent_edges = graph.edges_directed(node_index, Direction::Incoming);

        // Iterate over all parent edges and aggregate the scopes
        let scope = parent_edges
            .map(|e| {
                let mut parent_scope = ret[&e.source()].clone();
                parent_scope.extend(get_scope(e, &parent_scope));
                parent_scope
            })
            .reduce(&merge_scopes)
            .unwrap_or_default();

        ret.insert(node_index, scope);
    }
    ret
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

impl<K: IndexKey, P: 'static, I: Default> Default for AutomatonBuilder<K, P, I>
where
    Constraint<K, P>: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K: IndexKey, P: 'static, I: Default, M> FromIterator<Vec<Constraint<K, P>>>
    for AutomatonBuilder<K, P, I>
where
    Constraint<K, P>: Clone + Eq,
    I: IndexingScheme<BindMap = M>,
    M: BindMap<Key = K>,
{
    fn from_iter<T: IntoIterator<Item = Vec<Constraint<K, P>>>>(iter: T) -> Self {
        Self::from_constraints(iter)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use insta::assert_snapshot;
    use rstest::{fixture, rstest};
    use tests::modify::tests::{constraints, root_child, root_grandchildren};

    use super::modify::tests::{automaton, automaton2};
    use crate::{
        automaton::tests::TestAutomaton, constraint::tests::TestConstraint,
        constraint_tree::ConstraintTree, indexing::tests::TestIndexingScheme,
        predicate::tests::TestPredicate,
    };

    use super::*;

    // Dummy ToMutuallyExclusiveTree implementation for TestConstraint
    impl ToConstraintsTree<usize> for TestPredicate {
        fn to_constraints_tree(preds: Vec<TestConstraint>) -> ConstraintTree<TestConstraint> {
            // We take the first `k` constraints to be mutually exclusive,
            // where `k` is given by the arity of the first predicate (this has
            // no meaning).
            let (inds, preds): (Vec<_>, Vec<_>) = preds
                .into_iter()
                .enumerate()
                .sorted_by(|(_, p1), (_, p2)| p1.cmp(p2))
                .unzip();
            let Some(k) = preds.first().map(|c| c.predicate().arity) else {
                return ConstraintTree::new();
            };
            let first_k = preds.into_iter().take(k);
            let mut tree = ConstraintTree::new();
            let new_children = tree.add_children(tree.root(), first_k).collect_vec();
            for (index, child) in inds.into_iter().zip(new_children) {
                tree.add_constraint_index(child, index);
            }
            tree
        }
    }

    impl<K: IndexKey, P: 'static, I: Default> AutomatonBuilder<K, P, I> {
        pub(crate) fn into_matcher(self) -> ConstraintAutomaton<K, P, I> {
            ConstraintAutomaton {
                graph: self.graph.into_inner(),
                root: self.root,
                host_indexing: self.host_indexing,
            }
        }
    }

    impl<K: IndexKey, P: 'static, I: Default> ConstraintAutomaton<K, P, I> {
        pub(crate) fn wrap_builder(
            self,
            with_builder: impl Fn(&mut AutomatonBuilder<K, P, I>),
        ) -> Self {
            let mut builder = AutomatonBuilder::from_matcher(self);
            with_builder(&mut builder);
            builder.into_matcher()
        }
    }

    #[test]
    fn test_add_constraint_tree() {
        let mut builder = AutomatonBuilder::<_, _, TestIndexingScheme>::default();
        let n2 = builder.add_non_det_node();
        let constraint_tree = {
            let mut tree = ConstraintTree::new();
            let tree_child = tree.get_or_add_child(tree.root(), TestConstraint::new(vec![1]));
            tree.add_constraint_index(tree_child, 0);
            let tree_gchild = tree.get_or_add_child(tree_child, TestConstraint::new(vec![2]));
            tree.add_constraint_index(tree_gchild, 1);
            tree
        };
        builder.add_constraint_tree(constraint_tree, builder.root, &[n2, n2]);
        assert_snapshot!(builder.into_matcher().dot_string());
    }

    pub(crate) type TestBuilder = AutomatonBuilder<usize, TestPredicate, TestIndexingScheme>;

    #[test]
    fn test_build() {
        let mut builder = TestBuilder::new();
        let [constraint_root, _, constraint_b, constraint_c, constraint_d] = constraints();
        builder.add_pattern(vec![constraint_root.clone(), constraint_b.clone()], 0, None);
        builder.add_pattern(vec![constraint_root.clone(), constraint_c.clone()], 1, None);
        builder.add_pattern(vec![constraint_root, constraint_d.clone()], 2, None);
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
    fn test_make_det_noop(automaton: TestAutomaton) {
        let automaton2 = automaton.clone();
        let root_child = root_child(&automaton);
        let automaton = automaton.wrap_builder(|b| b.make_det(root_child));

        assert_eq!(automaton.graph.node_count(), automaton2.graph.node_count());
        assert_eq!(automaton.graph.edge_count(), automaton2.graph.edge_count());
    }

    #[rstest]
    fn test_make_det(automaton2: TestAutomaton) {
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
    fn test_make_unique(automaton2: TestAutomaton) {
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

    #[fixture]
    fn automaton3() -> TestAutomaton {
        let mut builder = TestBuilder::default();
        let n1 = builder.add_constraint(builder.root, TestConstraint::new(vec![0]));
        let n2 = builder.add_constraint(builder.root, TestConstraint::new(vec![1]));
        let f = builder.add_fail(n1);
        builder.append_edge(n2, f, None);
        builder.into_matcher()
    }

    #[rstest]
    fn test_mergeable_nodes(automaton3: TestAutomaton) {
        let builder = AutomatonBuilder::from_matcher(automaton3);
        let merge_nodes = builder.find_mergeable_states(NodeIndex::new(1).into());
        assert_eq!(
            merge_nodes,
            vec![NodeIndex::new(1).into(), NodeIndex::new(2).into()]
        );
    }
}
