use std::{borrow::Borrow, collections::VecDeque};

use itertools::Itertools;

use crate::branch_selector::EvaluateBranchSelector;
use crate::indexing::{bindings_hash, Binding, IndexedData};
use crate::{indexing::IndexKey, BindMap, HashMap, HashSet, PatternID};

use super::view::GraphView;
use super::{ConstraintAutomaton, StateID};

impl<K: IndexKey, B> ConstraintAutomaton<K, B> {
    /// Run the automaton on the `host` input data.
    pub fn run<'a, 'd, M, D>(&'a self, host: &'d D) -> AutomatonTraverser<'a, 'd, K, B, M, D>
    where
        M: Default,
    {
        AutomatonTraverser::new(self, host)
    }

    fn update_bindings<D: IndexedData<K>>(
        &self,
        state: StateID,
        bindings: D::BindMap,
        host: &D,
    ) -> Vec<D::BindMap> {
        // Try to bind all required keys
        let mut all_bindings = host.bind_all(bindings, self.min_scope(state).iter().copied());

        // Retain only the required bindings
        let required_bindings_set = self.max_scope(state);
        for bindings in all_bindings.iter_mut() {
            bindings.retain_keys(required_bindings_set);
        }

        all_bindings
    }

    /// An iterator of the allowed transitions
    fn next_legal_states<'a, D>(
        &'a self,
        state: TraversalState<D::BindMap>,
        host: &'a D,
    ) -> impl Iterator<Item = TraversalState<D::BindMap>> + 'a
    where
        D: IndexedData<K>,
        B: EvaluateBranchSelector<D, D::Value, Key = K>,
    {
        let Some(br) = self.branch_selector(state.state_id) else {
            return None.into_iter().flatten();
        };
        let reqs = br.required_bindings();

        let all_bindings = self.update_bindings(state.state_id, state.bindings, host);

        let all_next_states = all_bindings.into_iter().flat_map(move |bindings| {
            // Evaluate the predicates to find the allowed transitions
            let reqs_bindings = reqs
                .iter()
                .map(|k| match bindings.get_binding(k) {
                    Binding::Bound(v) => Some(v.borrow().clone()),
                    Binding::Failed => None,
                    Binding::Unbound => panic!(
                        "tried to use unbound key {k:?} in state {:?}",
                        state.state_id
                    ),
                })
                .collect_vec();
            let transitions = br.eval(&reqs_bindings, host);

            // Traverse allowed transitions to next states
            let bindings_clone = bindings.clone();
            let next_states = transitions.into_iter().map(move |index| {
                let state_id = self.nth_child(state.state_id, index);
                TraversalState {
                    state_id,
                    bindings: bindings_clone.clone(),
                }
            });

            // Always traverse the epsilon transition if it exists
            let epsilon_state = self
                .fail_child(state.state_id)
                .map(|state_id| TraversalState { state_id, bindings });

            next_states.chain(epsilon_state)
        });

        Some(all_next_states).into_iter().flatten()
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
pub struct AutomatonTraverser<'a, 'd, K: Ord, B, M, D> {
    matches_queue: VecDeque<(PatternID, M)>,
    state_queue: VecDeque<TraversalState<M>>,
    automaton: &'a ConstraintAutomaton<K, B>,
    host: &'d D,
    /// For each state, the set of hashes of bindings already visited (prune
    /// search if the same binding has been visited before)
    previous_bindings: HashMap<StateID, HashSet<u64>>,
}

impl<'a, 'd, K: IndexKey, B, M: Default, D> AutomatonTraverser<'a, 'd, K, B, M, D> {
    fn new(automaton: &'a ConstraintAutomaton<K, B>, host: &'d D) -> Self {
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

    /// Mark a state as visited if it wasn't visited already.
    ///
    /// Return whether the visit was successful, i.e. the state had not been
    /// visited before.
    ///
    /// Keep track of visited states by hashing the bindings on the max_scope
    /// set.
    fn visit(&mut self, state: &TraversalState<M>) -> bool
    where
        M: BindMap<Key = K>,
    {
        let state_id = state.state_id;
        let hash = bindings_hash(
            &state.bindings,
            self.automaton.max_scope(state_id).iter().copied(),
        );
        self.previous_bindings
            .entry(state_id)
            .or_default()
            .insert(hash)
    }

    fn complete_bindings(&self, bindings: D::BindMap, reqs: &[K]) -> Vec<D::BindMap>
    where
        D: IndexedData<K>,
    {
        if reqs.iter().any(|k| bindings.get_binding(k).is_failed()) {
            // Some of the required bindings could not be matched
            return vec![];
        }

        // Find keys that must still be bound
        let missing_keys = reqs
            .iter()
            .filter(|k| bindings.get_binding(k).is_unbound())
            .copied()
            .collect_vec();

        // Try to bind all required keys
        let mut all_bindings = if missing_keys.is_empty() {
            vec![bindings]
        } else {
            self.host.bind_all(bindings, missing_keys)
        };

        // Remove bindings that could not bind all required keys
        all_bindings
            .retain(|new_bindings| !reqs.iter().any(|k| new_bindings.get_binding(k).is_failed()));

        for new_bindings in all_bindings.iter_mut() {
            // Retain only the required bindings
            new_bindings.retain_keys(&reqs.iter().copied().collect())
        }

        all_bindings
    }
}

impl<B, D, K> Iterator for AutomatonTraverser<'_, '_, K, B, D::BindMap, D>
where
    K: IndexKey,
    D: IndexedData<K>,
    B: EvaluateBranchSelector<D, D::Value, Key = K>,
{
    type Item = (PatternID, D::BindMap);

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some(state) = self.state_queue.pop_front() else {
                break;
            };
            if !self.visit(&state) {
                continue;
            }

            // Add all matching patterns to the output queue
            for (&id, reqs) in self.automaton.matches(state.state_id) {
                let bindings = self.complete_bindings(state.bindings.clone(), reqs);
                self.matches_queue
                    .extend(bindings.into_iter().map(|bindings| (id, bindings)));
            }

            // Proceed to next states
            let legal_next_states = self.automaton.next_legal_states(state, self.host);
            self.state_queue.extend(legal_next_states);
        }
        self.matches_queue.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        automaton::{builder::tests::TestBuildConfig, tests::TestBuilder},
        constraint::tests::{TestConstraint, TestConstraintClass, TestPattern, TestPredicate},
        indexing::tests::TestData,
        HashSet,
    };

    fn true_constraint(cls: TestConstraintClass) -> TestConstraint {
        match cls {
            TestConstraintClass::One(_, _) => TestConstraint::new(TestPredicate::NotEqualOne),
            TestConstraintClass::Two(_, _) => TestConstraint::new(TestPredicate::AlwaysTrueTwo),
            TestConstraintClass::Three => TestConstraint::new(TestPredicate::AlwaysTrueThree),
        }
    }

    fn false_constraint(cls: TestConstraintClass) -> TestConstraint {
        match cls {
            TestConstraintClass::One(_, _) => TestConstraint::new(TestPredicate::AreEqualOne),
            TestConstraintClass::Two(_, _) => panic!("no always false constraint for two"),
            TestConstraintClass::Three => TestConstraint::new(TestPredicate::NeverTrueThree),
        }
    }

    #[test]
    fn run_automaton() {
        use TestConstraintClass::*;
        let one = One("", "");
        let two = Two("", "");
        let three = Three;

        let p1 = TestPattern::from_constraints(vec![true_constraint(one), false_constraint(three)]);

        let p2 = TestPattern::from_constraints(vec![true_constraint(one), true_constraint(two)]);

        let p3 = TestPattern::from_constraints(vec![
            true_constraint(one),
            true_constraint(two),
            true_constraint(three),
        ]);

        let p4 = TestPattern::from_constraints(vec![
            true_constraint(one),
            true_constraint(two),
            false_constraint(three),
        ]);

        let builder = TestBuilder::try_from_patterns([p1, p2, p3, p4], Default::default()).unwrap();
        let (automaton, ids) = builder.build(TestBuildConfig::default());
        let matches: HashSet<_> = automaton.run(&TestData).map(|(id, _)| id).collect();
        assert_eq!(matches, HashSet::from_iter([ids[1], ids[2]]));
    }
}
