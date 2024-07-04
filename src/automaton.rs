//! Core `ConstraintAutomaton` data structure and builder.
//!
//! Use [AutomatonBuilder] to construct an automaton from lists of constraints.

mod builder;
mod traversal;
mod view;

use petgraph::dot::Dot;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::stable_graph::{StableDiGraph, StableGraph};

use std::fmt::{self, Debug};

use derive_more::{From, Into};

use crate::PatternID;
pub use builder::AutomatonBuilder;

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
pub struct ConstraintAutomaton<C, I> {
    /// The transition graph
    graph: StableDiGraph<State, Transition<C>>,
    /// The root of the transition graph
    root: StateID,
    /// The indexing scheme used
    host_indexing: I,
}

impl<C, I: Default> Default for ConstraintAutomaton<C, I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C, I> ConstraintAutomaton<C, I> {
    /// An empty constraint automaton with default indexing scheme.
    ///
    /// Use the [AutomatonBuilder] to construct an automaton from a list of
    /// constraints.
    pub fn new() -> Self
    where
        I: Default,
    {
        Self::with_indexing_scheme(Default::default())
    }

    /// An empty constraint automaton, with the given indexing scheme
    pub fn with_indexing_scheme(host_indexing: I) -> Self {
        let mut graph = StableGraph::new();
        let root = graph.add_node(State::default());
        Self {
            graph,
            root: root.into(),
            host_indexing,
        }
    }

    /// Get its dot string representation
    pub fn dot_string(&self) -> String
    where
        C: Debug,
    {
        format!("{:?}", Dot::new(&self.graph))
    }

    /// Get the root state ID
    fn root(&self) -> StateID {
        self.root
    }

    /// Get the indexing scheme used by the automaton
    pub fn host_indexing(&self) -> &I {
        &self.host_indexing
    }
}

impl<C: Debug, I> Debug for ConstraintAutomaton<C, I> {
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
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct State {
    /// The pattern matches at current state
    matches: Vec<PatternID>,
    /// Whether the state is deterministic
    deterministic: bool,
    /// The order of the outgoing contraint transitions. Must map one-to-one to
    /// the outgoing constraint edges, i.e. the edges with non-None weights.
    constraint_order: Vec<TransitionID>,
    /// The order of the outgoing epsilon transitions. Must map one-to-one to
    /// the outgoing epsilon edges, i.e. the edges with None weights.
    epsilon_order: Vec<TransitionID>,
}

impl Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.deterministic {
            write!(f, "D")?;
        } else {
            write!(f, "ND")?;
        }
        if !self.matches.is_empty() {
            write!(f, " {:?}", self.matches)?;
        }
        Ok(())
    }
}

/// A transition from one state to another.
///
/// A transition has an optional constraint that must be satisfied for the
/// transition to occur. A None constraint is either an "epsilon" or "FAIL"
/// constraint, depending on the state's deterministic flag.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Transition<C> {
    constraint: Option<C>,
}

impl<C: Debug> Debug for Transition<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(constraint) = &self.constraint {
            write!(f, "{:?}", constraint)
        } else {
            write!(f, "ε")
        }
    }
}
