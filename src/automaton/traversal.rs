use std::collections::VecDeque;

use itertools::Itertools;
use portgraph::{LinkView, PortGraph, PortView};

use crate::{
    constraint::{ScopeConstraint, ScopeMap},
    BiMap, PatternID, Symbol,
};

use super::{OutPort, ScopeAutomaton, StateID};

impl<C: ScopeConstraint + Clone> ScopeAutomaton<C> {
    pub fn run<'a>(&'a self, root: C::Value, graph: C::DataRef<'a>) -> Traverser<'a, C> {
        Traverser::new(self, root, graph)
    }

    /// An iterator of the allowed transitions
    fn legal_transitions<'a, 's: 'a>(
        &'a self,
        state: StateID,
        ass: &'a TraverseState<C>,
        data: C::DataRef<'s>,
    ) -> impl Iterator<Item = (OutPort, ScopeMap<C>)> + 'a {
        self.outputs(state).filter_map(move |edge| {
            Some((edge, self.predicate(edge).is_satisfied(&ass.map, data)?))
        })
    }

    fn is_valid_assignment(&self, state: StateID, ass: &TraverseState<C>) -> bool {
        if state != ass.state_id {
            return false;
        }
        let ass = &ass.map;
        self.scope(state).iter().all(|s| ass.contains_left(s))
    }
}

/// An iterator for traversing a scope automaton
pub struct Traverser<'a, C: ScopeConstraint> {
    matches_queue: VecDeque<PatternID>,
    state_queue: VecDeque<TraverseState<C>>,
    automaton: &'a ScopeAutomaton<C>,
    data: C::DataRef<'a>,
}

impl<'a, C: ScopeConstraint + Clone> Traverser<'a, C> {
    fn new(automaton: &'a ScopeAutomaton<C>, root: C::Value, data: C::DataRef<'a>) -> Self {
        let ass = TraverseState::new(automaton.root, root);
        if !automaton.is_valid_assignment(automaton.root, &ass) {
            panic!("Input is not a valid assignment of root scope");
        }
        let matches_queue = VecDeque::new();
        let state_queue = VecDeque::from_iter([ass]);
        Self {
            matches_queue,
            state_queue,
            automaton,
            data,
        }
    }

    fn enqueue_state(
        &mut self,
        out_port: OutPort,
        mut ass: TraverseState<C>,
        new_symb: Option<(C::Symbol, C::Value)>,
    ) {
        let next_state = next_state(&self.automaton.graph, out_port);
        let next_scope = self.automaton.scope(next_state);
        ass.state_id = next_state;
        ass.map.retain(|s, _| next_scope.contains(s));
        if let Some((symb, val)) = new_symb {
            let failed = ass.map.insert(symb, val).did_overwrite();
            if failed {
                panic!("Tried to overwrite in assignment map");
            }
        }
        self.state_queue.push_back(ass)
    }
}

impl<'a, C: ScopeConstraint + Clone> Iterator for Traverser<'a, C> {
    type Item = PatternID;

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some(ass) = self.state_queue.pop_front() else {
                break;
            };
            let state = ass.state_id;
            self.matches_queue.extend(self.automaton.matches(state));
            let mut transitions = self.automaton.legal_transitions(state, &ass, self.data);
            if self.automaton.is_deterministic(state) {
                if let Some((edge, new_symbs)) = transitions.next() {
                    drop(transitions);
                    for (s, v) in new_symbs {
                        self.enqueue_state(edge, ass.clone(), Some((s, v)));
                    }
                }
            } else {
                for (edge, new_symbs) in transitions.collect_vec() {
                    for (s, v) in new_symbs {
                        self.enqueue_state(edge, ass.clone(), Some((s, v)));
                    }
                }
            }
        }
        self.matches_queue.pop_front()
    }
}

/// A map of scope symbols to values in the universe
///
/// For now, all scope assignments must be bijective, i.e. each value has
/// at most one symbol it is assigned to.
///
/// ## Type parameters
/// - A: A function from symbols to values
#[derive(Clone)]
struct TraverseState<C: ScopeConstraint> {
    map: ScopeMap<C>,
    state_id: StateID,
}

impl<C: ScopeConstraint> TraverseState<C> {
    fn new(root_state: StateID, root: C::Value) -> Self {
        let map = BiMap::from_iter([(C::Symbol::root(), root)]);
        Self {
            map,
            state_id: root_state,
        }
    }
}

/// Follow edge from an OutPort to the next state
fn next_state(g: &PortGraph, edge: OutPort) -> StateID {
    let OutPort(out_node, out_offset) = edge;
    let out_port_index = g
        .output(out_node.into(), out_offset)
        .expect("invalid OutPort");
    let in_port_index = g.port_link(out_port_index).expect("invalid transition");
    g.port_node(in_port_index)
        .expect("invalid port index")
        .into()
}
