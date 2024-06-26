use std::collections::VecDeque;
use std::fmt::Debug;

use itertools::Itertools;

use crate::{predicate::PredicateSatisfied, EdgeProperty, NodeProperty, PatternID, Universe};

use super::{AssignMap, OutPort, ScopeAutomaton, StateID, Symbol};

impl<PNode: NodeProperty, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    pub fn run<'s, U: Universe + 's>(
        &'s self,
        root: U,
        node_prop: impl for<'a> Fn(U, &'a PNode) -> bool + 's,
        edge_prop: impl for<'a> Fn(U, &'a PEdge) -> Vec<Option<U>> + 's,
    ) -> impl Iterator<Item = PatternID> + 's {
        let ass = AssignMap::new(self.root, root);
        if !self.is_valid_assignment(self.root, &ass) {
            panic!("Input is not a valid assignment of root scope");
        }
        Traverser::new(self, ass, node_prop, edge_prop)
    }

    /// An iterator of the allowed transitions
    fn legal_transitions<U: Universe>(
        &self,
        state: StateID,
        ass: &AssignMap<U>,
        node_prop: impl for<'a> Fn(U, &'a PNode) -> bool,
        edge_prop: impl for<'a> Fn(U, &'a PEdge) -> Vec<Option<U>>,
    ) -> Vec<(OutPort, Option<(Symbol, U)>)> {
        let mut predicate_results = self
            .out_ports(state)
            .map(move |edge| {
                (
                    edge,
                    self.predicate(edge)
                        .is_satisfied(&ass.map, &node_prop, &edge_prop),
                )
            })
            .collect_vec();
        if self.is_deterministic(state) {
            let mut det_predicate_results = Vec::new();
            // All predicate results are broadcast to have length `n_opts`
            if let Some(n_opts) = broadcast(&mut predicate_results) {
                // For all 0 <= i < n_opts, find the first edge with a satisfied predicate
                for i in 0..n_opts {
                    let (edge, res) = predicate_results
                        .iter()
                        .map(|(edge, all_res)| (*edge, all_res.get(i).expect("invalid broadcast")))
                        .find_or_last(|&(_, res)| res != &PredicateSatisfied::No)
                        .expect("n_opts != None so predicates_results.len() > 0");
                    if let Some(action) = res.to_option() {
                        det_predicate_results.push((edge, action));
                    }
                }
            }
            det_predicate_results
        } else {
            // All satisfied predicates are considered
            predicate_results
                .into_iter()
                .flat_map(|(edge, all_res)| {
                    all_res
                        .into_iter()
                        .filter_map(move |res| res.to_option().map(|action| (edge, action)))
                })
                .collect()
        }
    }

    fn is_valid_assignment<U: Universe>(&self, state: StateID, ass: &AssignMap<U>) -> bool {
        if state != ass.state_id {
            return false;
        }
        let ass = &ass.map;
        self.scope(state).iter().all(|s| ass.contains_left(s))
    }
}

/// An iterator for traversing a scope automaton
///
/// ## Type parameters
///  - U: the universe that the symbols in scope get interpreted to
///  - A: scope assignments mapping symbols to value in U
///  - M: Markers
///  - Symb: the symbol type used to refer to values in scope
///  - P: a predicate to check for allowed transitions
///  - SU: scope updater, to modify scope as we transition along edges
///  - D: arbitrary input data that the automaton can refer to
struct Traverser<Map, A, FnN, FnE> {
    matches_queue: VecDeque<PatternID>,
    state_queue: VecDeque<(StateID, Map)>,
    automaton: A,
    node_prop: FnN,
    edge_prop: FnE,
}

impl<'a, Map, PNode, PEdge: EdgeProperty, FnN, FnE>
    Traverser<Map, &'a ScopeAutomaton<PNode, PEdge>, FnN, FnE>
{
    fn new(
        automaton: &'a ScopeAutomaton<PNode, PEdge>,
        ass: Map,
        node_prop: FnN,
        edge_prop: FnE,
    ) -> Self {
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([(automaton.root, ass)]);
        Self {
            matches_queue,
            state_queue,
            automaton,
            node_prop,
            edge_prop,
        }
    }
}

impl<'a, U: Universe, PNode: NodeProperty, PEdge: EdgeProperty, FnN, FnE>
    Traverser<AssignMap<U>, &'a ScopeAutomaton<PNode, PEdge>, FnN, FnE>
{
    fn enqueue_state(
        &mut self,
        out_port: OutPort,
        mut ass: AssignMap<U>,
        new_symb: Option<(Symbol, U)>,
    ) {
        let next_state = self.automaton.next_state(out_port);
        let next_scope = self.automaton.scope(next_state);
        ass.state_id = next_state;
        ass.map.retain(|s, _| next_scope.contains(s));
        if let Some((symb, val)) = new_symb {
            let failed = ass.map.insert(symb, val).did_overwrite();
            if failed {
                panic!("Tried to overwrite in assignment map");
            }
        }
        self.state_queue.push_back((next_state, ass))
    }
}

impl<'a, U: Universe, PNode, PEdge: EdgeProperty, FnN, FnE> Iterator
    for Traverser<AssignMap<U>, &'a ScopeAutomaton<PNode, PEdge>, FnN, FnE>
where
    PNode: NodeProperty,
    FnN: for<'b> Fn(U, &'b PNode) -> bool,
    FnE: for<'b> Fn(U, &'b PEdge) -> Vec<Option<U>>,
{
    type Item = PatternID;

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some((state, ass)) = self.state_queue.pop_front() else {
                break;
            };
            self.matches_queue.extend(self.automaton.matches(state));
            let transitions =
                self.automaton
                    .legal_transitions(state, &ass, &self.node_prop, &self.edge_prop);
            for (edge, new_symb) in transitions {
                self.enqueue_state(edge, ass.clone(), new_symb);
            }
        }
        self.matches_queue.pop_front()
    }
}

fn broadcast<V: Debug, U: Eq + Clone>(
    vec: &mut [(V, Vec<PredicateSatisfied<U>>)],
) -> Option<usize> {
    let Some(mut len) = vec
        .iter()
        .find(|(_, res)| {
            // Singleton `Yes` predicates can be broadcast
            res != &vec![PredicateSatisfied::Yes]
        })
        .map(|(_, res)| res.len())
    else {
        return vec.first().map(|(_, res)| res.len());
    };
    if len < 1 {
        // All non-broadcastable vecs are empty
        // There might still be a `Yes` predicate that we cannot get rid of,
        // so we broadcast all vecs to length 1
        len = 1;
        for (_, res) in vec.iter_mut() {
            if res.is_empty() {
                *res = vec![PredicateSatisfied::No];
            }
        }
    } else {
        // Broadcast all vecs to length `len`
        for (_, res) in vec.iter_mut() {
            if res.len() != len {
                if *res != vec![PredicateSatisfied::Yes] {
                    panic!(
                        "Could not broadcast predicate outputs. Are they all of the same length?"
                    );
                }
                *res = vec![PredicateSatisfied::Yes; len];
            }
        }
    }
    Some(len)
}
