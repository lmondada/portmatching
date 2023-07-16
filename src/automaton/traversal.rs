use std::collections::VecDeque;

use portgraph::{LinkView, PortGraph, PortView};

use crate::{predicate::PredicateSatisfied, EdgeProperty, PatternID, Universe};

use super::{AssignMap, OutPort, ScopeAutomaton, StateID, Symbol};

impl<PNode: Copy, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    pub fn run<'s, U: Universe + 's>(
        &'s self,
        root: U,
        node_prop: impl Fn(U, PNode) -> bool + 's,
        edge_prop: impl Fn(U, PEdge) -> Option<U> + 's,
    ) -> impl Iterator<Item = PatternID> + 's {
        let ass = AssignMap::new(self.root, root);
        if !self.is_valid_assignment(self.root, &ass) {
            panic!("Input is not a valid assignment of root scope");
        }
        Traverser::new(self, ass, node_prop, edge_prop)
    }

    /// An iterator of the allowed transitions
    fn legal_transitions<'s, U: Universe>(
        &'s self,
        state: StateID,
        ass: &'s AssignMap<U>,
        node_prop: impl Fn(U, PNode) -> bool + 's,
        edge_prop: impl Fn(U, PEdge) -> Option<U> + 's,
    ) -> impl Iterator<Item = (OutPort, Option<(Symbol, U)>)> + 's {
        self.outputs(state).filter_map(move |edge| {
            match self
                .predicate(edge)
                .is_satisfied(&ass.map, &node_prop, &edge_prop)
            {
                PredicateSatisfied::NewSymbol(new_symb, new_u) => {
                    (edge, Some((new_symb, new_u))).into()
                }
                PredicateSatisfied::Yes => (edge, None).into(),
                PredicateSatisfied::No => None,
            }
        })
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

fn enqueue_state<U: Universe>(
    queue: &mut VecDeque<(StateID, AssignMap<U>)>,
    node: StateID,
    mut ass: AssignMap<U>,
    new_symb: Option<(Symbol, U)>,
) {
    if let Some((symb, val)) = new_symb {
        let failed = ass.map.insert(symb, val).did_overwrite();
        if failed {
            panic!("Tried to overwrite in assignment map");
        }
    }
    queue.push_back((node, ass))
}

impl<'a, U: Universe, PNode, PEdge: EdgeProperty, FnN, FnE> Iterator
    for Traverser<AssignMap<U>, &'a ScopeAutomaton<PNode, PEdge>, FnN, FnE>
where
    PNode: Copy,
    PEdge: Copy,
    FnN: Fn(U, PNode) -> bool,
    FnE: Fn(U, PEdge) -> Option<U>,
{
    type Item = PatternID;

    fn next(&mut self) -> Option<Self::Item> {
        while self.matches_queue.is_empty() {
            let Some((state, ass)) = self.state_queue.pop_front() else {
                break
            };
            self.matches_queue.extend(self.automaton.matches(state));
            let mut transitions =
                self.automaton
                    .legal_transitions(state, &ass, &self.node_prop, &self.edge_prop);
            if self.automaton.is_deterministic(state) {
                if let Some((edge, new_symb)) = transitions.next() {
                    drop(transitions);
                    enqueue_state(
                        &mut self.state_queue,
                        next_state(&self.automaton.graph, edge),
                        ass,
                        new_symb,
                    );
                }
            } else {
                for (edge, new_symb) in transitions {
                    enqueue_state(
                        &mut self.state_queue,
                        next_state(&self.automaton.graph, edge),
                        ass.clone(),
                        new_symb,
                    );
                }
            }
        }
        self.matches_queue.pop_front()
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
