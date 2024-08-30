use std::collections::VecDeque;

use itertools::Itertools;
use petgraph::graph::NodeIndex;

use crate::{
    indexing::{self, IndexKey},
    utils, IndexMap, IndexingScheme, PatternID, Predicate,
};

use super::{ConstraintAutomaton, StateID};

impl<K, P, I> ConstraintAutomaton<K, P, I> {
    /// Run the automaton on the `host` input data.
    pub fn run<'a, 'd, D>(&'a self, host: &'d D) -> AutomatonTraverser<'a, 'd, K, P, I, D>
    where
        P: Predicate<D>,
        I: IndexingScheme<D>,
        I::Map: IndexMap<Key = K, Value = P::Value>,
    {
        AutomatonTraverser::new(self, host)
    }

    /// An iterator of the allowed transitions
    fn next_legal_states<'a, D, S>(
        &'a self,
        state: TraversalState<S>,
        host: &'a D,
    ) -> impl Iterator<Item = TraversalState<S>> + 'a
    where
        P: Predicate<D>,
        S: IndexMap<Key = K, Value = P::Value>,
        I: IndexingScheme<D, Map = S>,
        S: 'a,
        K: IndexKey,
        P::Value: Clone,
    {
        let non_fails = self
            .all_constraint_transitions(state.state_id)
            .map(|t| (self.next_state(t), self.constraint(t).unwrap()))
            .collect_vec();
        let required_keys = non_fails
            .iter()
            .flat_map(|(_, constraint)| constraint.required_bindings())
            .copied()
            .unique()
            .collect_vec();

        let mut all_bindings = self
            .host_indexing
            .try_bind_all(state.bindings.clone(), required_keys, host)
            .unwrap();

        if all_bindings.is_empty() {
            all_bindings.push(state.bindings);
        }

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
pub struct AutomatonTraverser<'a, 'd, K, P, I: IndexingScheme<D>, D> {
    matches_queue: VecDeque<(PatternID, I::Map)>,
    state_queue: VecDeque<TraversalState<I::Map>>,
    automaton: &'a ConstraintAutomaton<K, P, I>,
    host: &'d D,
}

impl<'a, 'd, K, P, I: IndexingScheme<D>, D> AutomatonTraverser<'a, 'd, K, P, I, D> {
    fn new(automaton: &'a ConstraintAutomaton<K, P, I>, host: &'d D) -> Self {
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([TraversalState::new(automaton.root())]);
        Self {
            matches_queue,
            state_queue,
            automaton,
            host,
        }
    }
}

impl<'a, 'd, P, D, I> Iterator for AutomatonTraverser<'a, 'd, indexing::Key<I, D>, P, I, D>
where
    P: Predicate<D>,
    I: IndexingScheme<D>,
    I::Map: IndexMap<Value = P::Value>,
{
    type Item = (PatternID, I::Map);

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some(state) = self.state_queue.pop_front() else {
                break;
            };
            self.matches_queue.extend(
                self.automaton
                    .matches(state.state_id)
                    .iter()
                    .map(|&id| (id, state.bindings.clone())),
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

    pub(super) fn next<K, P, I>(
        &mut self,
        automaton: &ConstraintAutomaton<K, P, I>,
    ) -> Option<StateID> {
        self.traverser.next(&automaton.graph).map(StateID)
    }
}

impl<K, P, I> ConstraintAutomaton<K, P, I> {
    pub(crate) fn toposort(&self) -> OnlineToposort {
        OnlineToposort::new(self.root())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constraint::tests::TestConstraint, indexing::tests::TestIndexingScheme, HashMap, HashSet,
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
        let mut automaton = ConstraintAutomaton::<TestConstraint, TestIndexingScheme>::new();
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
                    &(),
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
                    &(),
                )
                .map(|TraversalState { state_id, .. }| state_id),
        );
        // Deterministic, so only true child should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child]));
    }

    #[test]
    fn legal_transitions_all_false() {
        let mut automaton = ConstraintAutomaton::<TestConstraint, TestIndexingScheme>::new();
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
                    &(),
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
                    &(),
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
                    &(),
                )
                .count(),
            1
        );
    }

    #[test]
    fn run_automaton() {
        let mut automaton = ConstraintAutomaton::<TestConstraint, TestIndexingScheme>::new();
        automaton.add_pattern(vec![true_constraint(), false_constraint()], PatternID(0));
        let match_pattern1 = PatternID(1);
        automaton.add_pattern(vec![true_constraint(), true_constraint()], match_pattern1);
        let match_pattern2 = PatternID(2);
        automaton.add_pattern(
            vec![true_constraint(), true_constraint(), true_constraint()],
            match_pattern2,
        );
        automaton.add_pattern(
            vec![true_constraint(), true_constraint(), false_constraint()],
            PatternID(3),
        );
        let matches: HashSet<_> = automaton.run(&()).map(|(id, _)| id).collect();
        assert_eq!(
            matches,
            HashSet::from_iter([match_pattern1, match_pattern2])
        );
    }
}
