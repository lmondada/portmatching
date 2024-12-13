use std::collections::{BTreeMap, BTreeSet, VecDeque};

use itertools::Itertools;
use petgraph::acyclic::Acyclic;

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    branch_selector::{BranchSelector, CreateBranchSelector},
    indexing::IndexKey,
    pattern::{Pattern, Satisfiable},
    BindMap, HashSet, IndexingScheme, PatternID, PatternLogic,
};

mod modify;

use super::{view::GraphView, BuildConfig, State, TransitionGraph};

/// Create constraint automata from lists of patterns, given by lists of
/// constraints.
pub struct AutomatonBuilder<PT, K: Ord, B> {
    /// The matcher being built, made of a graph, a root state and an indexing scheme
    graph: Acyclic<TransitionGraph<K, B>>,
    /// The root of the graph
    root: StateID,
    /// The list of all patterns, in order of addition
    patterns: Vec<PT>,
    /// The scopes that each pattern requires
    pattern_scopes: Vec<HashSet<K>>,
    /// A map of all created states, to be reused when possible
    hashcons: BTreeMap<BTreeSet<(PatternID, PT)>, StateID>,
}

impl<PT, K: IndexKey, B> AutomatonBuilder<PT, K, B> {
    /// Construct an empty automaton builder.
    pub fn new() -> Self {
        let mut graph = TransitionGraph::new();
        let root = graph.add_node(State::default());
        Self {
            graph: Acyclic::try_from_graph(graph).unwrap(),
            root: root.into(),
            patterns: Vec::new(),
            pattern_scopes: Vec::new(),
            hashcons: BTreeMap::new(),
        }
    }

    fn get_scope(&self, pattern: PatternID) -> &HashSet<K> {
        &self.pattern_scopes[pattern.0]
    }

    fn populate_max_scope(&mut self, state: StateID, patterns: impl IntoIterator<Item = PatternID>)
    where
        K: IndexKey,
    {
        let mut max_scope = BTreeSet::default();

        // Add keys that are required for any of the current matches
        for keys in self.matches(state).values() {
            max_scope.extend(keys);
        }

        // Add keys that are required for any of the future patterns
        for p in patterns {
            max_scope.extend(self.get_scope(p));
        }

        self.set_max_scope(state, max_scope);
    }

    fn known_bindings(&self, state: StateID) -> Vec<K> {
        // The known bindings of a state are the intersection of all min_scopes
        // of the parent states
        let mut known_bindings: Option<Vec<_>> = None;
        for parent in self.parents(state) {
            let parent_min_scope = self.min_scope(parent);
            if let Some(known_bindings) = known_bindings.as_mut() {
                known_bindings.retain(|b| parent_min_scope.contains(b));
            } else {
                known_bindings = Some(parent_min_scope.to_vec());
            }
        }
        known_bindings.unwrap_or_default()
    }
}

impl<PT, K: IndexKey, B> AutomatonBuilder<PT, K, B> {
    /// Construct an automaton builder from a list of patterns.
    pub fn from_patterns(
        patterns: impl IntoIterator<Item = impl Pattern<Key = K, Logic = PT>>,
    ) -> Self {
        let (patterns, pattern_scopes) = patterns
            .into_iter()
            .map(|p| {
                let scope = HashSet::from_iter(p.required_bindings());
                (p.into_logic(), scope)
            })
            .unzip();
        Self {
            patterns,
            pattern_scopes,
            ..Self::new()
        }
    }

    /// Get the root of the automaton.
    pub(super) fn root(&self) -> StateID {
        self.root
    }
}

/// State of the builder at one particular state in the graph
struct BuildState<P: PatternLogic> {
    state: StateID,
    patterns: BTreeSet<(PatternID, P)>,
    satisfied_constraints: BTreeSet<P::Constraint>,
}

impl<P: PatternLogic, K: IndexKey, B> AutomatonBuilder<P, K, B>
where
    B: CreateBranchSelector<P::Constraint, Key = K>,
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
    pub fn build<M>(
        mut self,
        config: BuildConfig<impl IndexingScheme<BindMap = M>>,
    ) -> (ConstraintAutomaton<K, B>, Vec<PatternID>)
    where
        M: BindMap<Key = K>,
    {
        // Turn patterns into a transition graph
        let pattern_ids = self.construct_graph();

        // Compute the minimum scopes for each state.
        self.populate_min_scopes(config.indexing_scheme);

        let matcher = ConstraintAutomaton {
            graph: self.graph.into_inner(),
            root: self.root,
        };
        (matcher, pattern_ids)
    }

    /// The main automaton construction function
    ///
    /// Given patterns, construct its constraint automaton.
    fn construct_graph(&mut self) -> Vec<PatternID> {
        // Drain all patterns into set
        let patterns: BTreeSet<_> = self
            .patterns
            .drain(..)
            .enumerate()
            .map(|(id, p)| (PatternID(id), p))
            .collect();
        let pattern_ids = patterns.iter().map(|(id, _)| *id).collect();

        // We process states in a FIFO order, is any other ordering better?
        let mut bfs_queue = VecDeque::new();

        // Register all patterns at the root
        self.hashcons.insert(patterns.clone(), self.root);
        bfs_queue.push_back(BuildState {
            state: self.root,
            patterns,
            satisfied_constraints: Default::default(),
        });

        while let Some(BuildState {
            state,
            patterns,
            satisfied_constraints,
        }) = bfs_queue.pop_front()
        {
            // Before proceeding, add matches to the current state
            let (completed_patterns, patterns) = partition_completed_patterns(patterns);
            self.add_matches(state, completed_patterns.into_iter().map(|(id, _)| id));

            if patterns.is_empty() {
                continue;
            }

            // Also, populate max scope, as it requires the whole set of
            // patterns and this info is lost later on
            self.populate_max_scope(state, patterns.iter().map(|&(id, _)| id));

            // Get the predicate class that makes us progress the most
            let cls = best_branch_class(patterns.iter().map(|(_, p)| p)).unwrap();

            // Now collect the predicates within the class. These are all
            // mutually exclusive by definition!
            let mut constraints = group_by_constraint(patterns, &cls, &satisfied_constraints);

            // Remove predicates with no patterns
            constraints.retain(|_, patterns| {
                drop_impossible_patterns(patterns);
                !patterns.is_empty()
            });

            // Branch selector will evaluate all predicates at once to find
            // which ones are satisfied
            let non_fail_predicates = constraints.keys().filter_map(|p| p.as_ref()).cloned();
            self.set_branch_selector(
                state,
                B::create_branch_selector(non_fail_predicates.collect()),
            );

            // For each predicate, create a new state and add it to the queue
            for (constraint, patterns) in constraints {
                let next_state = self.add_child(state, patterns.clone(), constraint.is_none());
                let mut satisfied_constraints = satisfied_constraints.clone();
                if let Some(constraint) = constraint {
                    satisfied_constraints.insert(constraint);
                }
                let build_state = BuildState {
                    state: next_state,
                    patterns,
                    satisfied_constraints,
                };
                bfs_queue.push_back(build_state);
            }
        }

        pattern_ids
    }

    /// Populate the scope of each automaton state.
    ///
    /// The scope is the set of bindings that are "relevant" to the current
    /// traversal. They are the set of bindings that
    ///  - are in all paths from root to `state`, and
    ///  - appear in at least one pattern match in the future of `state`, or
    ///  - are required to evaluate outgoing constraints at `state`.
    fn populate_min_scopes<M>(&mut self, indexing: impl IndexingScheme<BindMap = M>)
    where
        M: BindMap<Key = K>,
    {
        for state in self.all_states().collect_vec() {
            let Some(br) = self.branch_selector(state) else {
                continue;
            };
            let reqs = br.required_bindings().iter().copied();
            let known_bindings = self.known_bindings(state);
            let min_scope = indexing.all_missing_bindings(reqs, known_bindings);
            self.set_min_scope(state, min_scope);
        }
    }
}

fn group_by_constraint<P: PatternLogic>(
    patterns: BTreeSet<(PatternID, P)>,
    cls: &P::BranchClass,
    satisfied_constraints: &BTreeSet<P::Constraint>,
) -> BTreeMap<Option<P::Constraint>, BTreeSet<(PatternID, P)>> {
    let mut constraints: BTreeMap<Option<P::Constraint>, BTreeSet<(PatternID, P)>> =
        BTreeMap::default();
    for (id, pattern) in patterns {
        for (in_cls, other_constraints) in pattern.condition_on(cls, satisfied_constraints) {
            constraints
                .entry(in_cls)
                .or_default()
                .insert((id, other_constraints));
        }
    }
    constraints
}

fn approx_isize(f: f64) -> isize {
    (f * 10000.) as isize
}

fn best_branch_class<'p, P: PatternLogic + 'p>(
    patterns: impl IntoIterator<Item = &'p P>,
) -> Option<P::BranchClass> {
    // Collect all classes that are relevant to at least one pattern
    let classes: BTreeMap<P::BranchClass, f64> =
        patterns
            .into_iter()
            .fold(BTreeMap::default(), |mut classes, pattern| {
                for (cls, rank) in pattern.get_branch_classes() {
                    *classes.entry(cls).or_default() += rank;
                }
                classes
            });

    // The sum of the ranks is the expected number of patterns that will
    // be selected. Select the class that minimizes it.
    let min_rank = classes
        .into_iter()
        // We approximate with an int so that we get an Ord
        .min_by_key(|(_, rank)| approx_isize(*rank))?
        .0;
    Some(min_rank)
}

fn partition_completed_patterns<P: PatternLogic>(
    patterns: BTreeSet<(PatternID, P)>,
) -> (BTreeSet<(PatternID, P)>, BTreeSet<(PatternID, P)>) {
    patterns
        .into_iter()
        .partition(|(_, p)| matches!(p.is_satisfiable(), Satisfiable::Tautology))
}

fn drop_impossible_patterns<P: PatternLogic>(patterns: &mut BTreeSet<(PatternID, P)>) {
    patterns.retain(|(_, p)| !matches!(p.is_satisfiable(), Satisfiable::No))
}

impl<P, K: IndexKey, B> Default for AutomatonBuilder<P, K, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<PT: Pattern, B> FromIterator<PT> for AutomatonBuilder<PT::Logic, PT::Key, B> {
    fn from_iter<T: IntoIterator<Item = PT>>(iter: T) -> Self {
        Self::from_patterns(iter)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use rstest::{fixture, rstest};
    use tests::modify::tests::{constraints, root_child, root_grandchildren};

    use super::modify::tests::{automaton, automaton2};
    use crate::{
        automaton::tests::TestAutomaton,
        branch_selector::tests::TestBranchSelector,
        constraint::tests::TestConstraint,
        indexing::tests::TestStrIndexingScheme,
        predicate::{
            tests::{TestKey, TestPattern},
            PredicatePattern,
        },
    };

    use super::*;

    impl<P, B, K: Ord> AutomatonBuilder<P, K, B> {
        pub(crate) fn into_matcher(self) -> ConstraintAutomaton<K, B> {
            ConstraintAutomaton {
                graph: self.graph.into_inner(),
                root: self.root,
            }
        }
    }

    impl<K: IndexKey, B> ConstraintAutomaton<K, B> {
        pub(crate) fn wrap_builder<P: PatternLogic>(
            self,
            with_builder: impl Fn(&mut AutomatonBuilder<P, K, B>),
        ) -> Self {
            let mut builder = AutomatonBuilder {
                graph: Acyclic::try_from_graph(self.graph).unwrap(),
                root: self.root,
                ..AutomatonBuilder::new()
            };
            with_builder(&mut builder);
            builder.into_matcher()
        }
    }

    pub(crate) type TestBuilder = AutomatonBuilder<TestPattern, TestKey, TestBranchSelector>;
    pub(crate) type TestBuildConfig = BuildConfig<TestStrIndexingScheme>;

    #[test]
    fn test_build() {
        let c = PredicatePattern::from_constraints(constraints());
        let builder = TestBuilder::from_patterns([c]);
        let (matcher, pattern_ids) = builder.build(BuildConfig::<TestStrIndexingScheme>::default());
        assert_eq!(matcher.graph.node_count(), 6);
        assert_eq!(pattern_ids, vec![PatternID(0), PatternID(1), PatternID(2)]);
        let _ = matcher.children(matcher.root()).exactly_one().ok().unwrap();

        insta::assert_snapshot!(matcher.dot_string());
    }

    #[rstest]
    fn test_make_det_noop(automaton: TestAutomaton) {
        let automaton2 = automaton.clone();
        let root_child = root_child(&automaton);
        todo!("make_det")
        // let automaton = automaton.wrap_builder(|b| b.make_det(root_child));

        // assert_eq!(automaton.graph.node_count(), automaton2.graph.node_count());
        // assert_eq!(automaton.graph.edge_count(), automaton2.graph.edge_count());
    }

    #[rstest]
    fn test_make_det(automaton2: TestAutomaton) {
        let x_child = root_child(&automaton2);
        let [a_child, b_child, c_child, d_child] = root_grandchildren(&automaton2);
        todo!("make_det")

        // // Add a FAIL transition from x_child to a new state
        // let fail_child = automaton2.fail_child(x_child).unwrap();
        // // Add a common constraint to the fail child and a_child
        // let common_constraint = TestConstraint::new(vec![7, 8]);
        // let post_fail = automaton2
        //     .constraint_next_state(fail_child, &common_constraint)
        //     .unwrap();

        // let automaton2 = automaton2.wrap_builder(|b| b.make_det(x_child));

        // // Now `common_constraint` should be on all children, pointing to
        // // `post_fail`. For b, c, d, this should be the only transition.
        // for child in [b_child, c_child, d_child] {
        //     assert_eq!(automaton2.all_transitions(child).count(), 1);
        //     let transition = automaton2.all_transitions(child).next().unwrap();
        //     assert_eq!(automaton2.constraint(transition), Some(&common_constraint));
        //     assert_eq!(automaton2.next_state(transition), post_fail);
        // }
        // {
        //     // For a_child, there are two transitions with the same constraint
        //     assert_eq!(automaton2.all_transitions(a_child).count(), 2);
        //     let (t1, t2) = automaton2.all_transitions(a_child).collect_tuple().unwrap();
        //     assert_eq!(automaton2.constraint(t1), Some(&common_constraint));
        //     assert_eq!(automaton2.constraint(t2), Some(&common_constraint));
        // }
    }
}
