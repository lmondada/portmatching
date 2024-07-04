use std::collections::VecDeque;

use itertools::Itertools;
use petgraph::graph::NodeIndex;

use crate::{
    indexing::IndexKey, utils, Constraint, IndexMap, IndexingScheme, PatternID, Predicate,
};

use super::{AssignMap, ConstraintAutomaton, StateID, TransitionID};

impl<K: IndexKey, P: Eq + Clone, I> ConstraintAutomaton<Constraint<K, P>, I> {
    /// Run the automaton on the `host` input data.
    pub fn run<'a, 'd, D>(
        &'a self,
        host: &'d D,
    ) -> AutomatonTraverser<'a, 'd, AssignMap<K, P::Value>, Constraint<K, P>, I, D>
    where
        P: Predicate<D>,
    {
        AutomatonTraverser::new(self, host)
    }

    /// An iterator of the allowed transitions
    fn legal_transitions<'a, D, S>(
        &'a self,
        state: StateID,
        host: &'a D,
        known_bindings: S,
    ) -> impl Iterator<Item = (TransitionID, S)> + 'a
    where
        P: Predicate<D>,
        S: IndexMap<K, P::Value>,
        I: IndexingScheme<D, P::Value, Key = K>,
        S: 'a,
        P::Value: Clone,
    {
        let non_fails = self
            .transitions(state)
            .filter_map(|id| Some((id, self.constraint(id)?)))
            .collect_vec();
        let required_keys = non_fails
            .iter()
            .flat_map(|(_, constraint)| constraint.required_bindings())
            .copied()
            .unique()
            .collect_vec();

        let mut all_bindings = self
            .host_indexing
            .try_bind_all(known_bindings.clone(), required_keys, host)
            .unwrap();

        if all_bindings.is_empty() {
            all_bindings.push(known_bindings);
        }

        all_bindings.into_iter().flat_map(move |bindings| {
            let bindings_clone = bindings.clone();
            let mut valid_non_fails = non_fails
                .clone()
                .into_iter()
                .filter_map(move |(id, constraint)| {
                    let is_satisfied = constraint.is_satisfied(host, &bindings).unwrap_or(false);
                    is_satisfied.then_some((id, bindings.clone()))
                })
                .peekable();
            let is_empty = valid_non_fails.peek().is_none();
            let needs_fail_transition = !self.is_deterministic(state) || is_empty;
            let fail_transition = if needs_fail_transition {
                self.find_fail_transition(state)
            } else {
                None
            };
            valid_non_fails.chain(fail_transition.map(move |id| (id, bindings_clone)))
        })
    }
}

/// An iterator for traversing a constraint automaton
///
/// ## Type parameters
///  - S: scope assignments mapping variable names to values
///  - C: constraint type on transitions
///  - D: arbitrary input "host" data to evaluate constraints on.
pub struct AutomatonTraverser<'a, 'd, S, C, I, D> {
    matches_queue: VecDeque<(PatternID, S)>,
    state_queue: VecDeque<(StateID, S)>,
    automaton: &'a ConstraintAutomaton<C, I>,
    host: &'d D,
}

impl<'a, 'd, S: Default, C, I, D> AutomatonTraverser<'a, 'd, S, C, I, D> {
    fn new(automaton: &'a ConstraintAutomaton<C, I>, host: &'d D) -> Self {
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([(automaton.root, S::default())]);
        Self {
            matches_queue,
            state_queue,
            automaton,
            host,
        }
    }
}

impl<'a, 'd, S, K, P, D, I> Iterator for AutomatonTraverser<'a, 'd, S, Constraint<K, P>, I, D>
where
    I: IndexingScheme<D, P::Value, Key = K>,
    P: Predicate<D> + Eq + Clone,
    S: IndexMap<K, P::Value>,
    K: IndexKey,
    P::Value: Clone,
{
    type Item = (PatternID, S);

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some((state, known_bindings)) = self.state_queue.pop_front() else {
                break;
            };
            self.matches_queue.extend(
                self.automaton
                    .matches(state)
                    .iter()
                    .map(|&id| (id, known_bindings.clone())),
            );
            let transitions = self
                .automaton
                .legal_transitions(state, self.host, known_bindings);
            for (transition, new_ass) in transitions {
                let next_state = self.automaton.next_state(transition);
                self.state_queue.push_back((next_state, new_ass))
            }
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

    pub(super) fn next<C, I>(&mut self, automaton: &ConstraintAutomaton<C, I>) -> Option<StateID> {
        self.traverser.next(&automaton.graph).map(StateID)
    }
}

impl<K, P, I> ConstraintAutomaton<Constraint<K, P>, I> {
    pub(crate) fn toposort(&self) -> OnlineToposort {
        OnlineToposort::new(self.root())
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint::tests::TestConstraint, indexing::tests::TestIndexingScheme, HashSet};

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
        let true_child = automaton.add_transition(automaton.root(), Some(true_constraint()));
        automaton.add_transition(automaton.root(), Some(false_constraint()));
        let fail_child = automaton.add_transition(automaton.root(), None);
        let ass = AssignMap::default();
        let transitions = HashSet::from_iter(
            automaton
                .legal_transitions(automaton.root(), &(), ass.clone())
                .map(|(transition, _)| automaton.next_state(transition)),
        );
        // Non-deterministic, so both true and none children should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child, fail_child]));

        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        let transitions = HashSet::from_iter(
            automaton
                .legal_transitions(automaton.root(), &(), ass)
                .map(|(transition, _)| automaton.next_state(transition)),
        );
        // Deterministic, so only true child should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child]));
    }

    #[test]
    fn legal_transitions_all_false() {
        let mut automaton = ConstraintAutomaton::<TestConstraint, TestIndexingScheme>::new();
        automaton.add_transition(automaton.root(), Some(false_constraint()));
        let ass = AssignMap::default();
        // Only a False transition, so no legal moves
        assert_eq!(
            automaton
                .legal_transitions(automaton.root(), &(), ass.clone())
                .count(),
            0
        );

        // Add an epsilon transition
        automaton.add_transition(automaton.root(), None);
        // Now there is a valid move
        assert_eq!(
            automaton
                .legal_transitions(automaton.root(), &(), ass.clone())
                .count(),
            1
        );
        // Still valid when deterministic
        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        assert_eq!(
            automaton
                .legal_transitions(automaton.root(), &(), ass)
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
