use std::collections::{BTreeMap, BTreeSet, VecDeque};

use itertools::{izip, Itertools};
use petgraph::acyclic::Acyclic;

use crate::{
    automaton::{ConstraintAutomaton, StateID},
    branch_selector::CreateBranchSelector,
    indexing::IndexKey,
    pattern::{Pattern, Satisfiable},
    HashSet, IndexingScheme, PatternID, PatternLogic,
};

mod modify;

use super::{view::GraphView, BuildConfig, State, TransitionGraph};

type StateHashKey<PT> = (BTreeSet<(PatternID, PT)>, BTreeSet<PatternID>);

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
    hashcons: BTreeMap<StateHashKey<PT>, StateID>,
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
    pub fn build(
        mut self,
        config: BuildConfig<impl IndexingScheme<Key = K>>,
    ) -> (ConstraintAutomaton<K, B>, Vec<PatternID>) {
        // Turn patterns into a transition graph
        let pattern_ids = self.construct_graph();

        // Compute the minimum scopes for each state.
        self.populate_min_scopes(config.indexing_scheme);

        // Compute the maximum scopes (must happen after min scopes).
        self.populate_max_scope();

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
        // Find matches at the root
        let (completed_patterns, patterns) = partition_completed_patterns(patterns);
        let root_matches = BTreeSet::from_iter(completed_patterns.into_iter().map(|(id, _)| id));
        self.add_matches(self.root, root_matches.iter().copied());
        self.hashcons
            .insert((patterns.clone(), root_matches), self.root);

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
            if patterns.is_empty() {
                continue;
            }

            // 1. Vote: get the branch class that makes us progress the most
            let cls = select_best_class(patterns.iter().map(|(_, p)| p)).unwrap();

            // 2. Collect all constraints of interest to the patterns
            let nominations = patterns.iter().map(|(_, p)| p.nominate(&cls)).collect_vec();
            let mut nominations_iter = nominations.iter();
            // Take apart the patterns that do not nominate any constraints
            let (patterns, skip_patterns): (BTreeSet<_>, BTreeSet<_>) = patterns
                .into_iter()
                .partition(|_| !nominations_iter.next().unwrap().is_empty());
            let mut constraints = nominations.into_iter().flatten().collect_vec();
            constraints.sort_unstable();
            constraints.dedup();

            // 3. Apply all transition to each pattern. Track new patterns and
            // new matches
            let (mut next_patterns, mut next_matches) =
                apply_transitions(patterns, &constraints, &satisfied_constraints);

            // Remove transitions that do not lead to a new child
            retain_non_empty(&mut constraints, &mut next_patterns, &mut next_matches);

            // Add new children and queue them up when useful
            for (constraint, patterns, matches) in izip!(&constraints, next_patterns, next_matches)
            {
                let Some(next_state) = self.add_child(state, patterns.clone(), matches, false)
                else {
                    // The child state was already added
                    continue;
                };
                if !patterns.is_empty() {
                    let mut satisfied_constraints = satisfied_constraints.clone();
                    satisfied_constraints.insert(constraint.clone());
                    let build_state = BuildState {
                        state: next_state,
                        patterns,
                        satisfied_constraints,
                    };
                    bfs_queue.push_back(build_state);
                }
            }

            // Create the branch selector
            let br = B::create_branch_selector(constraints);
            self.set_branch_selector(state, br);

            // Add epsilon transition to skip patterns
            if !skip_patterns.is_empty() {
                if let Some(state) =
                    self.add_child(state, skip_patterns.clone(), BTreeSet::default(), true)
                {
                    let build_state = BuildState {
                        state,
                        patterns: skip_patterns,
                        satisfied_constraints,
                    };
                    bfs_queue.push_back(build_state);
                }
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
    fn populate_min_scopes(&mut self, indexing: impl IndexingScheme<Key = K>) {
        for state in self.all_states().collect_vec() {
            let reqs = self.required_bindings(state);
            let known_bindings = self.known_bindings(state);
            let min_scope = indexing.all_missing_bindings(reqs, known_bindings);
            self.set_min_scope(state, min_scope);
        }
    }

    fn populate_max_scope(&mut self)
    where
        K: IndexKey,
    {
        // Propagate max_scope from leaves to root (reverse topological sort)
        let topo = self.graph.nodes_iter().collect_vec();

        for node in topo.into_iter().rev() {
            let state = StateID(node);

            // The max_scope is the union of:
            // 1. The min_scope of the state
            let mut max_scope = BTreeSet::from_iter(self.min_scope(state).iter().copied());

            // 2. the required bindings at the state
            max_scope.extend(self.required_bindings(state));

            // 3. The keys required for any of the state's matches
            for keys in self.matches(state).values() {
                max_scope.extend(keys);
            }

            // 4. The max_scopes of any of the state's children
            for child in self.children(state) {
                max_scope.extend(self.max_scope(child));
            }

            self.set_max_scope(state, max_scope);
        }
    }

    fn required_bindings(&self, state: StateID) -> impl Iterator<Item = K> + '_ {
        if let Some(br) = self.branch_selector(state) {
            Some(br.required_bindings().iter().copied())
                .into_iter()
                .flatten()
        } else {
            None.into_iter().flatten()
        }
    }
}

fn retain_non_empty<P: PatternLogic>(
    constraints: &mut Vec<<P as PatternLogic>::Constraint>,
    next_patterns: &mut Vec<BTreeSet<(PatternID, P)>>,
    next_matches: &mut Vec<BTreeSet<PatternID>>,
) {
    let retain_transitions = next_patterns
        .iter()
        .zip(&*next_matches)
        .map(|(p, m)| !p.is_empty() || !m.is_empty())
        .collect_vec();

    macro_rules! retain {
        ($vec:expr) => {
            let mut iter = retain_transitions.iter().copied();
            $vec.retain(|_| iter.next().unwrap());
        };
    }

    retain!(constraints);
    retain!(next_patterns);
    retain!(next_matches);
}

fn apply_transitions<P: PatternLogic>(
    patterns: BTreeSet<(PatternID, P)>,
    transitions: &[<P as PatternLogic>::Constraint],
    known_constraints: &BTreeSet<<P as PatternLogic>::Constraint>,
) -> (Vec<BTreeSet<(PatternID, P)>>, Vec<BTreeSet<PatternID>>) {
    let mut next_patterns = vec![BTreeSet::default(); transitions.len()];
    let mut next_matches = vec![BTreeSet::default(); transitions.len()];
    for (id, pattern) in patterns {
        let new_patterns = pattern.condition_on(transitions, known_constraints);
        assert_eq!(transitions.len(), new_patterns.len());
        for (i, sat_p) in new_patterns.into_iter().enumerate() {
            let next_patterns = &mut next_patterns[i];
            let next_matches = &mut next_matches[i];
            match sat_p {
                Satisfiable::Yes(p) => {
                    next_patterns.insert((id, p));
                }
                Satisfiable::No => {}
                Satisfiable::Tautology => {
                    next_matches.insert(id);
                }
            }
        }
    }
    (next_patterns, next_matches)
}

fn approx_isize(f: f64) -> isize {
    (f * 10000.) as isize
}

fn select_best_class<'p, P: PatternLogic + 'p>(
    patterns: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = &'p P>>,
) -> Option<P::BranchClass> {
    let patterns = patterns.into_iter();
    let n_patterns = patterns.len() as f64;

    // Collect all classes that are relevant to at least one pattern
    let classes: BTreeMap<P::BranchClass, f64> =
        patterns
            .into_iter()
            .fold(BTreeMap::default(), |mut classes, pattern| {
                for (cls, rank) in pattern.rank_classes() {
                    *classes.entry(cls).or_insert(n_patterns) += rank - 1.;
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

    use crate::{
        branch_selector::tests::TestBranchSelector,
        constraint::tests::TestConstraint,
        indexing::tests::TestStrIndexingScheme,
        predicate::{
            tests::{TestKey, TestLogic, TestPattern, TestPredicate},
            PredicateLogic,
        },
    };

    use super::*;

    impl<P, B, K: Ord> AutomatonBuilder<P, K, B> {
        #[allow(unused)]
        pub(crate) fn into_matcher(self) -> ConstraintAutomaton<K, B> {
            ConstraintAutomaton {
                graph: self.graph.into_inner(),
                root: self.root,
            }
        }
    }

    impl<K: IndexKey, B> ConstraintAutomaton<K, B> {
        #[allow(unused)]
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

    pub(crate) type TestBuilder = AutomatonBuilder<TestLogic, TestKey, TestBranchSelector>;
    pub(crate) type TestBuildConfig = BuildConfig<TestStrIndexingScheme>;

    #[test]
    fn test_build() {
        let p1: TestPattern = PredicateLogic::from_constraints([
            TestConstraint::new(TestPredicate::AreEqualOne),
            TestConstraint::new(TestPredicate::AreEqualTwo),
        ])
        .into();
        let p2: TestPattern = PredicateLogic::from_constraints([
            TestConstraint::new(TestPredicate::AreEqualOne),
            TestConstraint::new(TestPredicate::AlwaysTrueThree),
        ])
        .into();
        let builder = TestBuilder::from_patterns([p1, p2]);
        let (matcher, pattern_ids) = builder.build(BuildConfig::<TestStrIndexingScheme>::default());
        assert_eq!(matcher.graph.node_count(), 5);
        assert_eq!(pattern_ids, vec![PatternID(0), PatternID(1)]);
        let _ = matcher.children(matcher.root()).exactly_one().ok().unwrap();

        insta::assert_snapshot!(matcher.dot_string());
    }

    // #[rstest]
    // fn test_make_det(automaton2: TestAutomaton) {
    //     let x_child = root_child(&automaton2);
    //     let [a_child, b_child, c_child, d_child] = root_grandchildren(&automaton2);
    //     todo!("make_det")

    //     // Add a FAIL transition from x_child to a new state
    //     let fail_child = automaton2.fail_child(x_child).unwrap();
    //     // Add a common constraint to the fail child and a_child
    //     let common_constraint = TestConstraint::new(vec![7, 8]);
    //     let post_fail = automaton2
    //         .constraint_next_state(fail_child, &common_constraint)
    //         .unwrap();

    //     let automaton2 = automaton2.wrap_builder(|b| b.make_det(x_child));

    //     // Now `common_constraint` should be on all children, pointing to
    //     // `post_fail`. For b, c, d, this should be the only transition.
    //     for child in [b_child, c_child, d_child] {
    //         assert_eq!(automaton2.all_transitions(child).count(), 1);
    //         let transition = automaton2.all_transitions(child).next().unwrap();
    //         assert_eq!(automaton2.constraint(transition), Some(&common_constraint));
    //         assert_eq!(automaton2.next_state(transition), post_fail);
    //     }
    //     {
    //         // For a_child, there are two transitions with the same constraint
    //         assert_eq!(automaton2.all_transitions(a_child).count(), 2);
    //         let (t1, t2) = automaton2.all_transitions(a_child).collect_tuple().unwrap();
    //         assert_eq!(automaton2.constraint(t1), Some(&common_constraint));
    //         assert_eq!(automaton2.constraint(t2), Some(&common_constraint));
    //     }
    // }
}
