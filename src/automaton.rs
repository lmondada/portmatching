mod builder;
mod modify;
mod traversal;
mod view;

pub use builder::LineBuilder;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::Graph;

use std::fmt::Debug;

use derive_more::{From, Into};

use crate::predicate::{EdgePredicate, Symbol};
use crate::{BiMap, EdgeProperty, HashSet, PatternID, Universe};

/// A state ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateID(NodeIndex);

#[derive(Copy, Clone, Debug)]
struct OutPort {
    state: NodeIndex,
    position: usize,
}

/// A node in the automaton
///
/// Nodes have zero, one or many markers that are output when the state is traversed
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct State<PNode, PEdge, OffsetID> {
    /// The pattern matches at current state
    matches: Vec<PatternID>,
    /// The ordered transitions to next states
    predicates: Vec<(EdgePredicate<PNode, PEdge, OffsetID>, EdgeIndex)>,
    /// The scope of the state
    scope: HashSet<Symbol>,
    /// Whether the state is deterministic
    deterministic: bool,
}

/// An automaton-like datastructure that follows transitions based on input and state
///
/// ## Type parameters
/// - M: Markers
/// - Symb: Symbols in state scopes
/// - P: Predicates to determine allowable transitions
/// - SU: Functions that update scope at incoming ports
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeAutomaton<PNode, PEdge, OffsetID = <PEdge as EdgeProperty>::OffsetID> {
    graph: Graph<Option<State<PNode, PEdge, OffsetID>>, ()>,
    root: StateID,
}

impl<PNode: Clone, PEdge: EdgeProperty> Default for ScopeAutomaton<PNode, PEdge> {
    fn default() -> Self {
        let mut graph = Graph::new();
        let root = graph.add_node(Some(State {
            matches: Vec::new(),
            predicates: Vec::new(),
            scope: HashSet::from_iter([Symbol::root()]),
            deterministic: true,
        }));
        Self {
            graph,
            root: root.into(),
        }
    }
}

impl<PNode: Clone, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    /// A new scope automaton
    pub fn new() -> Self {
        Default::default()
    }

    /// Get its dot string representation
    pub fn dot_string(&self) -> String
    where
        PNode: Debug,
        PEdge: Debug,
        <PEdge as EdgeProperty>::OffsetID: Debug,
    {
        format!(
            "{:?}",
            Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
        )
    }

    pub(crate) fn root(&self) -> StateID {
        self.root
    }
}

/// A map of scope symbols to values in the universe
///
/// For now, all scope assignments must be bijective, i.e. each value has
/// at most one symbol it is assigned to.
///
/// ## Type parameters
/// - A: A function from symbols to values
#[derive(Clone, Debug)]
struct AssignMap<U: Universe> {
    map: BiMap<Symbol, U>,
    state_id: StateID,
}

impl<U: Universe> AssignMap<U> {
    fn new(root_state: StateID, root: U) -> Self {
        let map = BiMap::from_iter([(Symbol::root(), root)]);
        Self {
            map,
            state_id: root_state,
        }
    }
}
