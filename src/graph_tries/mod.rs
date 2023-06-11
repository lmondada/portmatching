//! Graph trie data structures.
//!
//! A graph trie is a data structure that stores a set of pattern graphs in a tree-like
//! structure (except that it has fallback edges, making it a directed acyclic graph).
//!
//! Traversing the trie from top to bottom along a path that is given
//! by the input graph yields all matches of the pattern graphs.
mod base;
mod build;
mod optimise;
#[doc(inline)]
pub use base::BaseGraphTrie;

use base::EdgeWeight;
use build::{trace_insert, GraphTrieBuilder};
use optimise::get_next_world_age;

use portgraph::{NodeIndex, PortGraph, PortIndex};

use crate::constraint::{Constraint, PortAddress};
// use crate::constraint::unweighted::{Constraint, PortAddress};

/// A state in a graph trie.
///
/// Graph tries are stored themselves as port graphs, so a state is just a
/// node index in a `PortGraph`.
pub type StateID = NodeIndex;
pub(crate) fn root_state() -> NodeIndex {
    NodeIndex::new(0)
}

type Graph<'g, C> = <C as Constraint>::Graph<'g>;

/// A graph trie.
///
/// The trie is stored as a port graph. Each state (node) of the trie has
/// an [`GraphTrie::address`], which makes it correspond to a vertex in the
/// input graph. This address is encoded using the state's [`GraphTrie::spine`].
///
/// To follow an edge transition from the current state, [`GraphTrie::port_offset`]
/// indicates the outgoing port that should be followed to the next node.
/// Which [`StateTransition`] that edge corresponds to defines which children
/// of the trie should be explored next.
///
/// States can be deterministic or non-deterministic. If it is deterministic,
/// then children are considered in order, and the first one that matches is
/// chosen. If it is non-deterministic, then all children for which the transition
/// conditions are satisfied must be considered.
pub trait GraphTrie {
    /// The type of the transition conditions.
    type Constraint;

    /// The addressing scheme used for the trie.
    type Address;

    /// The underlying graph structure of the trie.
    fn trie(&self) -> &PortGraph;

    /// The address of the current state
    fn port_address(&self, state: StateID) -> Option<&Self::Address>;

    /// The ports corresponding to the current `state`.
    fn ports<'g>(&self, state: StateID, graph: Graph<'g, Self::Constraint>) -> Vec<PortIndex>
    where
        Self::Constraint: Constraint,
        Self::Address: PortAddress<Graph<'g, Self::Constraint>>,
    {
        let out_port = self.port_address(state);
        out_port.map(|p| p.ports(graph)).unwrap_or_default()
    }

    /// The unique port corresponding to the current `state`, or `None`.
    fn port<'g>(&self, state: StateID, graph: Graph<'g, Self::Constraint>) -> Option<PortIndex>
    where
        Self::Constraint: Constraint,
        Self::Address: PortAddress<Graph<'g, Self::Constraint>>,
    {
        let out_port = self.port_address(state);
        out_port.and_then(|p| p.port(graph))
    }

    /// The transition condition for the child linked at `port`.
    ///
    /// `port` must be an outgoing port of the trie.
    fn transition(&self, port: PortIndex) -> Option<&Self::Constraint>;

    /// Whether the current state is not deterministic.
    fn is_non_deterministic(&self, state: StateID) -> bool;

    // /// The next node in the current graph if we follow one transition
    // fn next_node(&self, state: StateID, graph: &Graph<Self::Constraint>) -> Option<NodeIndex>
    // where
    //     Self::Constraint: Constraint,
    //     Self::Address: PortAddress<Graph<Self::Constraint>>,
    // {
    //     let port = self.port(state, graph)?;
    //     let in_port = graph.port_link(port)?;
    //     graph.port_node(in_port)
    // }

    // /// The next port in the current graph if we follow one transition
    // fn next_port_offset(
    //     &self,
    //     state: StateID,
    //     graph: &PortGraph,
    //     root: NodeIndex,
    // ) -> Option<PortOffset> {
    //     let ports = self.ports(state, graph, root);
    //     let port = if ports.len() == 1 {
    //         ports[0]
    //     } else {
    //         return None;
    //     };
    //     let in_port = graph.port_link(port)?;
    //     graph.port_offset(in_port)
    // }

    /// The transitions to follow from `state`
    fn get_transitions<'g>(
        &self,
        state: StateID,
        graph: Graph<'g, Self::Constraint>,
    ) -> Vec<PortIndex>
    where
        Self::Constraint: Constraint,
        Self::Address: PortAddress<Graph<'g, Self::Constraint>>,
        Graph<'g, Self::Constraint>: Copy,
    {
        // All transitions in `state` that are allowed for `graph`
        let mut transitions = self.trie().outputs(state).filter(move |&out_p| {
            let Some(port_addr) = self.port_address(state) else { return false };
            self.transition(out_p)
                .map(|c| c.is_satisfied(port_addr, graph))
                .unwrap_or(true)
        });
        if self.is_non_deterministic(state) {
            transitions.collect()
        } else {
            transitions.next().into_iter().collect()
        }
    }

    /// All allowed state transitions from `state`.
    fn next_states<'g>(&self, state: StateID, graph: Graph<'g, Self::Constraint>) -> Vec<StateID>
    where
        Self::Constraint: Constraint,
        Self::Address: PortAddress<Graph<'g, Self::Constraint>>,
        Graph<'g, Self::Constraint>: Copy,
    {
        self.get_transitions(state, graph)
            .into_iter()
            .filter_map(|out_p| {
                let in_p = self.trie().port_link(out_p)?;
                self.trie().port_node(in_p)
            })
            .collect()
    }
}
