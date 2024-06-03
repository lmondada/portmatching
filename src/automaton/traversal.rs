use std::collections::VecDeque;
use std::{fmt::Debug, hash::Hash};

use itertools::Itertools;

use crate::{
    ArityPredicate, AssignPredicate, Constraint, ConstraintType, FilterPredicate, PatternID,
    VariableNaming, VariableScope,
};

use super::{AssignMap, ConstraintAutomaton, StateID, TransitionID};

impl<V: VariableNaming, U: Debug, AP: ArityPredicate, FP: ArityPredicate>
    ConstraintAutomaton<Constraint<V, U, AP, FP>>
{
    pub(super) fn constraint_type(&self, state: StateID) -> Option<ConstraintType<V>> {
        let ct = self.constraints(state).next().map(|c| c.into())?;
        debug_assert!(self
            .constraints(state)
            .all(|c| ConstraintType::from(c) == ct));
        Some(ct)
    }

    /// Run the automaton on the `host` input data, given a binding for the root.
    pub fn run<'a, 'd, D>(
        &'a self,
        root_binding: U,
        host: &'d D,
    ) -> AutomatonTraverser<'a, 'd, AssignMap<V, U>, Constraint<V, U, AP, FP>, D>
    where
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
    {
        let ass = AssignMap::from_iter([(V::root_variable(), root_binding)]);
        AutomatonTraverser::new(self, host, ass)
    }

    /// An iterator of the allowed transitions
    fn legal_transitions<D, S>(&self, state: StateID, host: &D, ass: S) -> Vec<(TransitionID, S)>
    where
        U: Eq + Hash,
        S: VariableScope<V, U>,
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
    {
        let transitions = self.transitions(state);
        let mut new_scopes = transitions
            .filter_map(|transition| Some((transition, self.constraint(transition)?)))
            .flat_map(|(transition, constraint)| {
                constraint
                    .satisfy(host, ass.clone())
                    .unwrap()
                    .into_iter()
                    .map(move |new_ass| (transition, new_ass))
            });

        let det_transitions = match self.constraint_type(state) {
            Some(ConstraintType::Assign(var)) => {
                // We gather all mutually exclusive scope assignments
                let transitions = new_scopes.collect_vec();
                debug_assert!(
                    all_mutex_assignments(transitions.iter().map(|(_, s)| s), &var),
                    "Mutually exclusive constraints cannot suggest identical assignments"
                );
                transitions
            }
            Some(ConstraintType::Filter) => {
                // At most one filter constraint can be satisfied
                let transitions = new_scopes.next().into_iter().collect();
                debug_assert!(
                    new_scopes.next().is_none(),
                    "At most one filter constraint can be satisfied"
                );
                transitions
            }
            None => Vec::new(),
        };

        // Add FAIL or epsilon transition
        let Some(none_constraint) = self.find_constraint(state, None) else {
            return det_transitions;
        };
        if self.is_deterministic(state) {
            // Add FAIL transition if there is no other successful transition
            if det_transitions.is_empty() {
                vec![(none_constraint, ass)]
            } else {
                det_transitions
            }
        } else {
            // Add FAIL transition
            let mut transitions = det_transitions;
            transitions.push((none_constraint, ass));
            transitions
        }
    }
}

/// An iterator for traversing a constraint automaton
///
/// ## Type parameters
///  - S: scope assignments mapping variable names to values
///  - C: constraint type on transitions
///  - D: arbitrary input "host" data to evaluate constraints on.
pub struct AutomatonTraverser<'a, 'd, S, C, D> {
    matches_queue: VecDeque<PatternID>,
    state_queue: VecDeque<(StateID, S)>,
    automaton: &'a ConstraintAutomaton<C>,
    host: &'d D,
}

impl<'a, 'd, S, C, D> AutomatonTraverser<'a, 'd, S, C, D> {
    fn new(automaton: &'a ConstraintAutomaton<C>, host: &'d D, ass: S) -> Self {
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([(automaton.root, ass)]);
        Self {
            matches_queue,
            state_queue,
            automaton,
            host,
        }
    }
}

impl<'a, 'd, S, V, U, AP, FP, D> Iterator
    for AutomatonTraverser<'a, 'd, S, Constraint<V, U, AP, FP>, D>
where
    U: Eq + Hash + Debug,
    V: VariableNaming,
    S: VariableScope<V, U>,
    AP: AssignPredicate<U, D>,
    FP: FilterPredicate<U, D>,
{
    type Item = PatternID;

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some((state, ass)) = self.state_queue.pop_front() else {
                break;
            };
            self.matches_queue.extend(self.automaton.matches(state));
            let transitions = self.automaton.legal_transitions(state, self.host, ass);
            for (transition, new_ass) in transitions {
                let next_state = self.automaton.next_state(transition);
                self.matches_queue
                    .extend(self.automaton.matches(next_state));
                self.state_queue.push_back((next_state, new_ass))
            }
        }
        self.matches_queue.pop_front()
    }
}

/// Check if all the assignments of `var` are distinct
fn all_mutex_assignments<'s, V, U, S: 's>(
    scopes: impl Iterator<Item = &'s S> + ExactSizeIterator,
    var: &V,
) -> bool
where
    U: Eq + Hash,
    S: VariableScope<V, U>,
{
    let scopes = scopes.into_iter();
    let n_scopes = scopes.len();
    let n_unique = scopes.into_iter().flat_map(|s| s.get(var)).unique().count();
    n_unique == n_scopes
}

#[cfg(test)]
mod tests {
    use crate::{
        constraint,
        predicate::{tests::AssignEq, Predicate::Filter},
        HashSet,
    };

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ConstFilterPredicate(bool);
    impl ArityPredicate for ConstFilterPredicate {
        fn arity(&self) -> usize {
            0
        }
    }
    impl FilterPredicate<usize, ()> for ConstFilterPredicate {
        fn check(&self, _: &(), _: &[&usize]) -> bool {
            self.0
        }
    }
    type Constraint = constraint::Constraint<String, usize, AssignEq, ConstFilterPredicate>;

    fn true_constraint() -> Constraint {
        Constraint::try_new(Filter(ConstFilterPredicate(true)), vec![]).unwrap()
    }

    fn false_constraint() -> Constraint {
        Constraint::try_new(Filter(ConstFilterPredicate(false)), vec![]).unwrap()
    }

    #[test]
    fn legal_transitions() {
        let mut automaton = ConstraintAutomaton::<Constraint>::new();
        // Add a True, False and None constraint
        let true_child =
            automaton.add_transition_unknown_child(automaton.root(), Some(true_constraint()));
        automaton.add_transition_unknown_child(automaton.root(), Some(false_constraint()));
        let fail_child = automaton.add_transition_unknown_child(automaton.root(), None);
        let ass = AssignMap::default();
        let transitions = HashSet::from_iter(
            automaton
                .legal_transitions(automaton.root(), &(), ass.clone())
                .into_iter()
                .map(|(transition, _)| automaton.next_state(transition)),
        );
        // Non-deterministic, so both true and none children should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child, fail_child]));

        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        let transitions = HashSet::from_iter(
            automaton
                .legal_transitions(automaton.root(), &(), ass)
                .into_iter()
                .map(|(transition, _)| automaton.next_state(transition)),
        );
        // Deterministic, so only true child should be returned
        assert_eq!(transitions, HashSet::from_iter([true_child]));
    }

    #[test]
    fn legal_transitions_all_false() {
        let mut automaton = ConstraintAutomaton::<Constraint>::new();
        automaton.add_transition_unknown_child(automaton.root(), Some(false_constraint()));
        let ass = AssignMap::default();
        // Only a False transition, so no legal moves
        assert!(automaton
            .legal_transitions(automaton.root(), &(), ass.clone())
            .is_empty());

        // Add an epsilon transition
        automaton.add_transition_unknown_child(automaton.root(), None);
        // Now there is a valid move
        assert!(!automaton
            .legal_transitions(automaton.root(), &(), ass.clone())
            .is_empty());
        // Still valid when deterministic
        let root = automaton.root();
        automaton.graph[root.0].deterministic = true;
        assert!(!automaton
            .legal_transitions(automaton.root(), &(), ass)
            .is_empty());
    }

    #[test]
    fn run_automaton() {
        let mut automaton = ConstraintAutomaton::<Constraint>::new();
        automaton.add_constraints(vec![true_constraint(), false_constraint()]);
        let p2 = automaton.add_constraints(vec![true_constraint(), true_constraint()]);
        let p3 = automaton.add_constraints(vec![
            true_constraint(),
            true_constraint(),
            true_constraint(),
        ]);
        automaton.add_constraints(vec![
            true_constraint(),
            true_constraint(),
            false_constraint(),
        ]);
        let matches: HashSet<_> = automaton.run(0, &()).collect();
        assert_eq!(matches, HashSet::from_iter([p2, p3]));
    }
}
