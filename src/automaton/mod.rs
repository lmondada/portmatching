mod builders;
mod modify;
mod traversal;
mod view;

pub use builders::LineBuilder;

use std::{fmt::Debug, hash::Hash};

use derive_more::{From, Into};

use portgraph::dot::DotFormat;
use portgraph::{NodeIndex, PortGraph, PortMut, PortView, Weights};

use crate::constraint::{ScopeConstraint, ScopeMap};
use crate::{BiMap, HashSet, PatternID, Symbol};

/// A state ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StateID(NodeIndex);

/// the n-th outport of a state
#[derive(Clone, Copy, Debug)]
pub struct OutPort(StateID, usize);

impl OutPort {
    /// The state the outport belongs to
    #[allow(unused)]
    pub fn state(&self) -> StateID {
        self.0
    }
}

/// A node in the automaton
///
/// Nodes have zero, one or many markers that are output when the state is traversed
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct State<S: Hash + Eq> {
    matches: Vec<PatternID>,
    scope: HashSet<S>,
    deterministic: bool,
}

/// Weight of outgoing ports
///
/// Leaving a state, we need to satisfy the predicate P
#[derive(Clone, Debug, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Transition<C> {
    Constraint(C),
    Epsilon,
}

impl<C> From<Option<C>> for Transition<C> {
    fn from(c: Option<C>) -> Self {
        match c {
            Some(c) => Self::Constraint(c),
            None => Self::Epsilon,
        }
    }
}

impl<C: ScopeConstraint> Transition<C>
where
    C::Symbol: Clone,
    C::Value: Clone + Eq + Hash,
{
    fn is_satisfied(&self, scope: &ScopeMap<C>, graph: C::DataRef<'_>) -> Option<ScopeMap<C>> {
        match self {
            Self::Constraint(c) => c.is_satisfied(graph, scope),
            Self::Epsilon => Some((*scope).clone()),
        }
    }
}

// #[cfg(feature = "serde")]
// impl<PNode, PEdge> serde::Serialize for Transition<PNode, PEdge>
// where
//     PEdge: EdgeProperty + Debug,
//     PNode: Debug,
// {
//     fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
//         serializer.serialize_str(&format!("{:?}", self.predicate))
//     }
// }

// #[cfg(feature = "serde")]
// impl<'g, PNode, PEdge> serde::Deserialize<'g> for Transition<PNode, PEdge>
// where
//     PEdge: EdgeProperty + Debug,
//     PNode: Debug,
// {
//     fn deserialize<D: serde::Deserializer<'g>>(deserializer: D) -> Result<Self, D::Error> {
//         let s = String::deserialize(deserializer)?;
//         Ok(Self {
//             predicate: EdgePredicate::from_str(&s).map_err(serde::de::Error::custom)?,
//         })
//     }
// }

/// An automaton-like datastructure that follows transitions based on input and state
///
/// ## Type parameters
/// - M: Markers
/// - Symb: Symbols in state scopes
/// - P: Predicates to determine allowable transitions
/// - SU: Functions that update scope at incoming ports
#[derive(Clone, Debug)]
// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeAutomaton<C: ScopeConstraint> {
    graph: PortGraph,
    weights: Weights<Option<State<C::Symbol>>, Option<Transition<C>>>,
    root: StateID,
}

impl<C: ScopeConstraint + Clone> Default for ScopeAutomaton<C> {
    fn default() -> Self {
        let mut graph = PortGraph::new();
        let root: StateID = graph.add_node(0, 0).into();
        let weights = {
            let mut w = Weights::new();
            w[root.0] = Some(State {
                matches: Vec::new(),
                scope: HashSet::from_iter([C::Symbol::root()]),
                deterministic: true,
            });
            w
        };
        Self {
            graph,
            weights,
            root,
        }
    }
}

impl<C: ScopeConstraint + Clone> ScopeAutomaton<C> {
    /// A new scope automaton
    pub fn new() -> Self {
        Default::default()
    }

    pub(crate) fn str_weights(&self) -> Weights<String, String>
    where
        C: Debug,
        C::Symbol: Debug,
    {
        let mut str_weights = Weights::new();
        for n in self.graph.nodes_iter() {
            if let Some(w) = self.weights[n].as_ref() {
                str_weights[n] = format!("{:?}", w);
                if let Some(w) = self.weights[n].as_ref().map(|w| &w.matches) {
                    if !w.is_empty() {
                        str_weights[n] += &format!("[{:?}]", w);
                    }
                }
            }
        }
        for p in self.graph.ports_iter() {
            if let Some(w) = self.weights[p].as_ref() {
                str_weights[p] = format!("{:?}", w);
            }
        }
        str_weights
    }

    /// Get its dot string representation
    pub fn dot_string(&self) -> String
    where
        C: Debug,
        C::Symbol: Debug,
    {
        self.graph
            .dot_format()
            .with_weights(&self.str_weights())
            .finish()
    }

    pub(crate) fn root(&self) -> StateID {
        self.root
    }
}
