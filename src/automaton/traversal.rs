use std::hash::{Hash, Hasher};
use std::{borrow::Borrow, collections::VecDeque};

use itertools::Itertools;
use petgraph::graph::NodeIndex;
use rustc_hash::FxHasher;

use crate::indexing::{DataKVMap, IndexedData};
use crate::{
    indexing::{IndexKey, Key},
    utils, HashMap, HashSet, IndexMap, IndexingScheme, PatternID, Predicate,
};

use super::{ConstraintAutomaton, StateID};

impl<P, I: IndexingScheme> ConstraintAutomaton<Key<I>, P, I> {
    /// Run the automaton on the `host` input data.
    pub fn run<'a, 'd, D>(&'a self, host: &'d D) -> AutomatonTraverser<'a, 'd, Key<I>, P, I, D>
    where
        D: IndexedData<IndexingScheme = I>,
        P: Predicate<D>,
    {
        AutomatonTraverser::new(self, host)
    }

    /// An iterator of the allowed transitions
    fn next_legal_states<'a, D>(
        &'a self,
        state: TraversalState<DataKVMap<D>>,
        host: &'a D,
    ) -> impl Iterator<Item = TraversalState<DataKVMap<D>>> + 'a
    where
        P: Predicate<D>,
        D: IndexedData<IndexingScheme = I>,
    {
        // Try to bind all (as many as possible of the) required keys
        let mut all_bindings = host.bind_all(
            state.bindings,
            self.required_bindings(state.state_id).iter().copied(),
            true,
        );

        // Retain only the required bindings
        let required_bindings_set =
            HashSet::from_iter(self.required_bindings(state.state_id).to_vec());
        for bindings in all_bindings.iter_mut() {
            bindings.retain_keys(&required_bindings_set);
        }

        // Find all transitions that are satisfied
        // First non-fails, then fails (if needed)
        let non_fails = self
            .all_constraint_transitions(state.state_id)
            .map(|t| (self.next_state(t), self.constraint(t).unwrap()))
            .collect_vec();

        all_bindings.into_iter().flat_map(move |bindings| {
            let bindings_clone = bindings.clone();
            let mut valid_non_fails = non_fails
                .clone()
                .into_iter()
                .filter_map(move |(id, constraint)| {
                    let is_satisfied = constraint
                        .is_satisfied(host, &bindings_clone)
                        .unwrap_or(false);
                    if is_satisfied {
                        Some(TraversalState {
                            state_id: id,
                            bindings: bindings_clone.clone(),
                        })
                    } else {
                        None
                    }
                })
                .peekable();
            let is_empty = valid_non_fails.peek().is_none();
            let needs_fail_transition = !self.is_deterministic(state.state_id) || is_empty;
            let fail_transition = if needs_fail_transition {
                self.fail_next_state(state.state_id)
            } else {
                None
            };
            valid_non_fails
                .chain(fail_transition.map(|state_id| TraversalState { state_id, bindings }))
        })
    }
}

struct TraversalState<S> {
    state_id: StateID,
    bindings: S,
}

impl<S: Default> TraversalState<S> {
    fn new(state_id: StateID) -> Self {
        Self {
            state_id,
            bindings: S::default(),
        }
    }
}

/// An iterator for traversing a constraint automaton
///
/// ## Type parameters
///  - S: scope assignments mapping variable names to values
///  - C: constraint type on transitions
///  - D: arbitrary input "host" data to evaluate constraints on.
pub struct AutomatonTraverser<'a, 'd, K: IndexKey, P, I: IndexingScheme, D> {
    matches_queue: VecDeque<(PatternID, I::Map)>,
    state_queue: VecDeque<TraversalState<I::Map>>,
    automaton: &'a ConstraintAutomaton<K, P, I>,
    host: &'d D,
    /// For each state, the set of hashes of bindings already visited (prune
    /// search if the same binding has been visited before)
    previous_bindings: HashMap<StateID, HashSet<u64>>,
}

impl<'a, 'd, K: IndexKey, P, I: IndexingScheme, D> AutomatonTraverser<'a, 'd, K, P, I, D> {
    fn new(automaton: &'a ConstraintAutomaton<K, P, I>, host: &'d D) -> Self {
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([TraversalState::new(automaton.root())]);
        let previous_bindings = HashMap::default();
        Self {
            matches_queue,
            state_queue,
            automaton,
            host,
            previous_bindings,
        }
    }
}

impl<'a, 'd, P, I: IndexingScheme, D> AutomatonTraverser<'a, 'd, Key<I>, P, I, D> {
    /// Mark a state as visited if it wasn't visited already.
    ///
    /// Return whether the visit was successful, i.e. the state had not been
    /// visited before.
    ///
    /// Keep track of visited stated by hashing the bindings on the set of
    /// variables that are relevant, i.e.
    ///  - the state scope (relevant for future constraints)
    ///  - the bindings required by matches at the current state
    fn visit(&mut self, state: &TraversalState<I::Map>) -> bool {
        let state_id = state.state_id;
        let scope = &self.automaton.graph[state_id.0].required_bindings;
        let req_bindings_matches = self.automaton.matches(state_id).values().flatten().unique();
        let all_useful_bindings = scope.iter().chain(req_bindings_matches).copied();
        let hash = bindings_hash(&state.bindings, all_useful_bindings);
        self.previous_bindings
            .entry(state_id)
            .or_default()
            .insert(hash)
    }
}

fn bindings_hash<S: IndexMap>(bindings: &S, scope: impl IntoIterator<Item = S::Key>) -> u64 {
    let mut hasher = FxHasher::default();
    for key in scope {
        let value = bindings.get(&key);
        value.as_ref().map(|v| v.borrow()).hash(&mut hasher);
    }
    hasher.finish()
}

impl<'a, 'd, P, D> Iterator
    for AutomatonTraverser<'a, 'd, Key<D::IndexingScheme>, P, D::IndexingScheme, D>
where
    P: Predicate<D>,
    D: IndexedData,
{
    type Item = (PatternID, DataKVMap<D>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some(state) = self.state_queue.pop_front() else {
                break;
            };
            if !self.visit(&state) {
                continue;
            }
            self.matches_queue
                .extend(
                    self.automaton
                        .matches(state.state_id)
                        .iter()
                        .flat_map(|(&id, scope)| {
                            let new_keys = scope
                                .iter()
                                .copied()
                                .filter(|k| state.bindings.get(k).is_none())
                                .collect_vec();
                            let bindings = if !new_keys.is_empty() {
                                self.host.bind_all(state.bindings.clone(), new_keys, false)
                            } else {
                                vec![state.bindings.clone()]
                            };
                            bindings.into_iter().map(move |mut bindings| {
                                bindings.retain_keys(scope);
                                (id, bindings)
                            })
                        }),
                );
            let legal_next_states = self.automaton.next_legal_states(state, self.host);
            self.state_queue.extend(legal_next_states);
        }
        self.matches_queue.pop_front()
    }
}

pub(crate) struct OnlineToposort {
    traverser: utils::OnlineToposort<NodeIndex>,
}

impl OnlineToposort {
    fn new(root: StateID) -> Self {
        Self {
            traverser: utils::online_toposort(root.0),
        }
    }

    pub(super) fn next<K: IndexKey, P, I>(
        &mut self,
        automaton: &ConstraintAutomaton<K, P, I>,
    ) -> Option<StateID> {
        self.traverser.next(&automaton.graph).map(StateID)
    }
}

impl<K: IndexKey, P, I> ConstraintAutomaton<K, P, I> {
    pub(crate) fn toposort(&self) -> OnlineToposort {
        OnlineToposort::new(self.root())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        automaton::tests::TestAutomaton, constraint::tests::TestConstraint,
        indexing::tests::TestData, HashMap, HashSet,
    };

    use super::*;

    fn true_constraint() -> TestConstraint {
        TestConstraint::new(vec![])
    }

    fn false_constraint() -> TestConstraint {
        TestConstraint::new(vec![3, 4])
    }

    #[test]
    fn legal_transitions() {
        let mut automaton = TestAutomaton::new();
        // Add a True, False and None constraint
        let true_child = automaton.add_constraint(automaton.root(), true_constraint());
        automaton.add_constraint(automaton.root(), false_constraint());
        let fail_child = automaton.add_fail(automaton.root());
        let ass = HashMap::default();
        let transitions = HashSet::from_iter(
            automaton
                .next_legal_states(
                    TraversalState {
                        state_id: automaton.root(),
                        bindings: ass.clone(),
                    },
                    &TestData,
                )
                .map(|TraversalState { state_id, .. }| state_id),
        );
        // Non-deterministic, so both true and none children should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child, fail_child]));

        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        let transitions = HashSet::from_iter(
            automaton
                .next_legal_states(
                    TraversalState {
                        state_id: automaton.root(),
                        bindings: ass.clone(),
                    },
                    &TestData,
                )
                .map(|TraversalState { state_id, .. }| state_id),
        );
        // Deterministic, so only true child should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child]));
    }

    #[test]
    fn legal_transitions_all_false() {
        let mut automaton = TestAutomaton::new();
        automaton.add_constraint(automaton.root(), false_constraint());
        let ass = HashMap::default();
        // Only a False transition, so no legal moves
        assert_eq!(
            automaton
                .next_legal_states(
                    TraversalState {
                        state_id: automaton.root(),
                        bindings: ass.clone(),
                    },
                    &TestData,
                )
                .count(),
            0
        );

        // Add an epsilon transition
        automaton.add_fail(automaton.root());
        // Now there is a valid move
        assert_eq!(
            automaton
                .next_legal_states(
                    TraversalState {
                        state_id: automaton.root(),
                        bindings: ass.clone(),
                    },
                    &TestData,
                )
                .count(),
            1
        );
        // Still valid when deterministic
        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        assert_eq!(
            automaton
                .next_legal_states(
                    TraversalState {
                        state_id: automaton.root(),
                        bindings: ass.clone(),
                    },
                    &TestData,
                )
                .count(),
            1
        );
    }

    #[test]
    fn run_automaton() {
        let mut automaton = TestAutomaton::new();
        automaton.add_pattern(
            vec![true_constraint(), false_constraint()],
            PatternID(0),
            None,
        );
        let match_pattern1 = PatternID(1);
        automaton.add_pattern(
            vec![true_constraint(), true_constraint()],
            match_pattern1,
            None,
        );
        let match_pattern2 = PatternID(2);
        automaton.add_pattern(
            vec![true_constraint(), true_constraint(), true_constraint()],
            match_pattern2,
            None,
        );
        automaton.add_pattern(
            vec![true_constraint(), true_constraint(), false_constraint()],
            PatternID(3),
            None,
        );
        let matches: HashSet<_> = automaton.run(&TestData).map(|(id, _)| id).collect();
        assert_eq!(
            matches,
            HashSet::from_iter([match_pattern1, match_pattern2])
        );
    }
}
