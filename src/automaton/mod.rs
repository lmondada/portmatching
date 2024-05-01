mod builder;
mod modify;
mod traversal;
mod view;

pub use builder::AutomatonBuilder;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::{StableDiGraph, StableGraph};

use std::fmt::Debug;

use derive_more::{From, Into};

use crate::{HashMap, PatternID};

/// A state ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateID(NodeIndex);

/// A transition ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TransitionID(EdgeIndex);

/// A node in the automaton
///
/// Nodes have zero, one or many markers that are output when the state is traversed
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct State {
    /// The pattern matches at current state
    matches: Vec<PatternID>,
    /// Whether the state is deterministic
    deterministic: bool,
}

/// A transition from one state to another.
///
/// A transition has an optional constraint that must be satisfied for the
/// transition to occur. A None constraint is either an "epsilon" or "FAIL"
/// constraint, depending on the state's deterministic flag.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Transition<C> {
    constraint: Option<C>,
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
pub struct ScopeAutomaton<C> {
    graph: StableDiGraph<State, Transition<C>>,
    root: StateID,
    n_patterns: usize,
}

impl<C> Default for ScopeAutomaton<C> {
    fn default() -> Self {
        let mut graph = StableGraph::new();
        let root = graph.add_node(State {
            matches: Vec::new(),
            deterministic: true,
        });
        Self {
            graph,
            root: root.into(),
            n_patterns: 0,
        }
    }
}

impl<C> ScopeAutomaton<C> {
    /// A new scope automaton
    pub fn new() -> Self {
        Default::default()
    }

    /// Get its dot string representation
    pub fn dot_string(&self) -> String
    where
        C: Debug,
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

type AssignMap<V, U> = HashMap<V, U>;
