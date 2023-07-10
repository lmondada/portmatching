mod builders;
mod modify;
mod traversal;
mod view;

pub(crate) use builders::LineBuilder;

use std::fmt::Debug;



use derive_more::{From, Into};

use portgraph::dot::DotFormat;
use portgraph::{NodeIndex, PortGraph, PortMut, PortView, Weights};

use crate::predicate::{EdgePredicate, Symbol};
use crate::{BiMap, EdgeProperty, HashSet, PatternID, Universe};

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
struct State {
    matches: Vec<PatternID>,
    scope: HashSet<Symbol>,
    deterministic: bool,
}

/// Weight of outgoing ports
///
/// Leaving a state, we need to satisfy the predicate P
#[derive(Clone, Debug, Copy, From, Into)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Transition<PNode, PEdge, OffsetID> {
    predicate: EdgePredicate<PNode, PEdge, OffsetID>,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ScopeAutomaton<PNode, PEdge, OffsetID = <PEdge as EdgeProperty>::OffsetID> {
    graph: PortGraph,
    weights: Weights<Option<State>, Option<Transition<PNode, PEdge, OffsetID>>>,
    root: StateID,
}

impl<PNode: Copy, PEdge: EdgeProperty> ScopeAutomaton<PNode, PEdge> {
    /// A new scope automaton
    ///
    /// ## Parameters
    /// - root_scope: The scope of the root state
    pub fn new() -> Self {
        let mut graph = PortGraph::new();
        let root: StateID = graph.add_node(0, 0).into();
        let weights = {
            let mut w = Weights::new();
            w[root.0] = Some(State {
                matches: Vec::new(),
                scope: HashSet::from_iter([Symbol::root()]),
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

    pub(crate) fn str_weights(&self) -> Weights<String, String>
    where
        PNode: Debug,
        PEdge: Debug,
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
        PNode: Debug,
        PEdge: Debug,
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
