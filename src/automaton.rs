//! Core `ConstraintAutomaton` data structure and builder.
//!
//! Use [AutomatonBuilder] to construct an automaton from lists of constraints.

mod builder;
mod traversal;
mod view;

use derive_where::derive_where;
use petgraph::dot::Dot;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::StableDiGraph;

use crate::branch_selector::DisplayBranchSelector;
use crate::HashMap;
use std::collections::BTreeSet;
use std::fmt::{self, Debug};
use std::hash::Hash;

use derive_more::{From, Into};

use crate::indexing::IndexKey;
use crate::PatternID;
pub use builder::AutomatonBuilder;

/// The underlying petgraph type for the ConstraintAutomaton.
type TransitionGraph<K, B> = StableDiGraph<State<K, B>, ()>;

/// An automaton-like datastructure to evaluate sets of constraints efficiently.
///
/// Organises lists of constraints into an automaton, leading to performance
/// gains when many patterns (i.e. lists of constraints) are evaluated together.
/// The reduction in the number of constraints that must be evaluated (compared
/// to evaluating each constraint of each pattern individually) is achieved by
///   1. Ordering constraints and merging identical constraints across multiple
///      patterns,
///   2. Partition sets of constraints into mutually exclusive sets to exclude
///      non-matching patterns efficiently, and optionally break down complex
///      constraints into smaller atomic units,
///   3. Delaying in some cases the evaluation of constraints: if pattern P1 and
///      P2 evaluate constraints C1 and C2 respectively but C2 > C1, then we can
///      delay C2's evaluation for the case that it will be evaluated by P1 later on.
///
/// The input data that the automaton is run on is called the host data. Host data
/// is accessed by the automaton through an [crate::IndexingScheme].
///
/// ## Type parameters
/// - C: Constraint type on transitions
/// - I: Indexing scheme of the host data
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        deserialize = "K: Eq + Hash + serde::Deserialize<'de>, B: serde::Deserialize<'de>"
    ))
)]
pub struct ConstraintAutomaton<K: Ord, B> {
    /// The transition graph
    graph: TransitionGraph<K, B>,
    /// The root of the transition graph
    root: StateID,
}

impl<K: IndexKey, P> Default for ConstraintAutomaton<K, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for building an automaton
///
/// This struct holds configuration parameters used during the automaton construction process.
#[derive(Clone, Debug, Default)]
pub struct BuildConfig<I> {
    /// The indexing scheme to use for the automaton.
    /// This determines how to index into host data through variable names (keys).
    pub indexing_scheme: I,
}

impl<K: IndexKey, B> ConstraintAutomaton<K, B> {
    /// An empty constraint automaton.
    ///
    /// Use the [AutomatonBuilder] to construct an automaton from a list of
    /// constraints.
    pub fn new() -> Self {
        let mut graph = TransitionGraph::new();
        let root = graph.add_node(State::default());
        Self {
            graph,
            root: root.into(),
        }
    }

    /// Get its dot string representation
    pub fn dot_string(&self) -> String
    where
        K: IndexKey,
        B: DisplayBranchSelector,
    {
        let str_graph = self.graph.map(&fmt_node, &fmt_edge(&self.graph));
        format!("{}", Dot::new(&str_graph))
    }

    /// Get the root state ID
    fn root(&self) -> StateID {
        self.root
    }

    /// Get the number of states in the automaton
    pub fn n_states(&self) -> usize {
        self.graph.node_count()
    }
}

fn fmt_node<K: IndexKey, B: DisplayBranchSelector>(_: NodeIndex, weight: &State<K, B>) -> String {
    let br = weight
        .branch_selector
        .as_ref()
        .map(|br| br.fmt_class())
        .unwrap_or_default();
    let matches = weight
        .matches
        .iter()
        .map(|(id, bindings)| format!("{}: {:?}", id, bindings))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{br}\n{matches}")
}

fn fmt_edge<'t, K: IndexKey, B: DisplayBranchSelector + 't>(
    graph: &TransitionGraph<K, B>,
) -> impl Fn(EdgeIndex, &()) -> String + '_ {
    |edge, &()| {
        let edge_id = TransitionID(edge);
        let src = graph.edge_endpoints(edge).unwrap().0;
        let src_weight = graph.node_weight(src).unwrap();
        if let Some(pos) = src_weight.edge_order.iter().position(|&e| e == edge_id) {
            let br = src_weight
                .branch_selector
                .as_ref()
                .expect("non trivial transition but no branch selector");
            br.fmt_nth_constraint(pos).to_string()
        } else {
            "FAIL".to_string()
        }
    }
}

impl<K: IndexKey, P: DisplayBranchSelector> Debug for ConstraintAutomaton<K, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dot_string())
    }
}

/// A state ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct StateID(NodeIndex);

/// A transition ID in a scope automaton
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, From, Into, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct TransitionID(EdgeIndex);

/// A node in the automaton
///
/// Nodes have zero, one or many markers that are output when the state is traversed
#[derive(Clone)]
#[derive_where(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        deserialize = "K: Eq + Hash + serde::Deserialize<'de>, B: serde::Deserialize<'de>"
    ))
)]
struct State<K: Ord, B> {
    /// The pattern matches at current state, along with the bindings associated
    /// with each match.
    ///
    /// The requried bindings are stored topological order.
    matches: HashMap<PatternID, Vec<K>>,
    /// The branch selector for the state
    ///
    /// None if the state has no child
    branch_selector: Option<B>,
    /// The order of the outgoing contraint transitions. Must map one-to-one to
    /// the outgoing constraint edges, i.e. the edges with non-None weights.
    edge_order: Vec<TransitionID>,
    /// The order of the outgoing epsilon transitions. Must map one-to-one to
    /// the outgoing epsilon edges, i.e. the edges with None weights.
    epsilon: Option<TransitionID>,
    /// The set of keys that must have been bound to process this state.
    min_scope: Vec<K>,
    /// Any key outside of this set can be forgotten.
    ///
    /// The ordering of the keys is used for unique hashing.
    max_scope: BTreeSet<K>,
}

impl<K: IndexKey, B> Debug for State<K, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.matches.is_empty() {
            write!(f, " {:?}", self.matches)?;
        }
        writeln!(f)?;
        writeln!(f, "max_scope: {:?}", self.max_scope)?;
        writeln!(f, "min_scope: {:?}", self.min_scope)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    pub(crate) use super::builder::tests::TestBuilder;
}
