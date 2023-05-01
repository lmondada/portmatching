//! Graph trie data structures.
//!
//! A graph trie is a data structure that stores a set of pattern graphs in a tree-like
//! structure (except that it has fallback edges, making it a directed acyclic graph).
//!
//! Traversing the trie from top to bottom along a path that is given
//! by the input graph yields all matches of the pattern graphs.
mod base;
mod cached;
#[doc(inline)]
pub use base::BaseGraphTrie;

use std::fmt::{self, Debug, Display};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::addressing::{
    pg::AsPathOffset, Address, AddressCache, AsSpineID, SkeletonAddressing, SpineAddress,
};

/// A state transition in a graph trie.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StateTransition<Address> {
    /// The port exists and is linked to another port.
    ///
    /// The addresses are the possible addresses that the other node is allowed
    /// to have, the port offset is the offset of the port at the other node.
    Node(Vec<Address>, PortOffset),
    /// The port exists but is not linked to anything.
    NoLinkedNode,
    /// The port does not exist.
    FAIL,
}

impl<A> Default for StateTransition<A> {
    fn default() -> Self {
        Self::FAIL
    }
}

impl<A: Debug> Display for StateTransition<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateTransition::Node(addrs, port) => {
                let addrs: Vec<_> = addrs.iter().map(|addr| format!("{:?}", addr)).collect();
                write!(f, "{}[{:?}]", addrs.join("/"), port)
            }
            StateTransition::NoLinkedNode => write!(f, "dangling"),
            StateTransition::FAIL => write!(f, "fail"),
        }
    }
}

impl<A> StateTransition<A> {
    fn transition_type(&self) -> usize {
        match &self {
            StateTransition::Node(_, _) => 0,
            StateTransition::NoLinkedNode => 1,
            StateTransition::FAIL => 2,
        }
    }

    pub(crate) fn map<B, F: FnMut(A) -> B>(self, mut f: F) -> StateTransition<B> {
        match self {
            StateTransition::Node(addrs, port) => {
                let addrs = addrs.into_iter().map(move |addr| f(addr)).collect();
                StateTransition::Node(addrs, port)
            }
            StateTransition::NoLinkedNode => StateTransition::NoLinkedNode,
            StateTransition::FAIL => StateTransition::FAIL,
        }
    }
}

/// A state in a graph trie.
///
/// Graph tries are stored themselves as port graphs, so a state is just a
/// node index in a `PortGraph`.
pub type StateID = NodeIndex;
pub(crate) fn root_state() -> NodeIndex {
    NodeIndex::new(0)
}

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
pub trait GraphTrie
where
    for<'n> <<Self as GraphTrie>::SpineID as SpineAddress>::AsRef<'n>:
        Copy + AsSpineID + AsPathOffset + PartialEq,
{
    /// The addressing scheme used for the spine.
    type SpineID: SpineAddress;
    // where
    //     for<'n> <Self::SpineID as SpineAddress>::AsRef<'n>: Copy + AsSpineID;

    /// The addressing scheme used for the trie.
    type Addressing<'g, 'n>: SkeletonAddressing<'g, 'n, Self::SpineID> + Clone
    where
        Self::SpineID: 'n;

    /// The underlying graph structure of the trie.
    fn trie(&self) -> &PortGraph;
    /// The address of a trie state.
    fn address<'n>(
        &'n self,
        state: StateID,
    ) -> Option<Address<<Self::SpineID as SpineAddress>::AsRef<'n>>>;
    /// The spine of a trie state.
    ///
    /// Useful for the address encoding.
    fn spine<'n>(&'n self, state: StateID) -> Option<&'n Vec<Self::SpineID>>;
    /// The port offset for the transition from `state`.
    fn port_offset(&self, state: StateID) -> Option<PortOffset>;
    /// The transition condition for the child linked at `port`.
    ///
    /// `port` must be an outgoing port of the trie.
    fn transition<'g, 'n, 'm: 'n>(
        &'n self,
        port: PortIndex,
        addressing: &Self::Addressing<'g, 'm>,
    ) -> StateTransition<(
        Self::Addressing<'g, 'n>,
        Address<<Self::SpineID as SpineAddress>::AsRef<'n>>,
    )>;
    /// Whether the current state is not deterministic.
    fn is_non_deterministic(&self, state: StateID) -> bool;

    /// The node in the current graph at `state`
    fn node<'g, 'n, C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'n>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let addr = self.address(state)?;
        addressing.get_node(&addr, cache)
    }

    /// The port in the current graph at `state`
    fn port<'g, 'n, C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'n>,
        cache: &mut C,
    ) -> Option<PortIndex> {
        let offset = self.port_offset(state)?;
        addressing
            .graph()
            .port_index(self.node(state, addressing, cache)?, offset)
    }

    /// The next node in the current graph if we follow one transition
    fn next_node<'g, 'n, C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'n>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let in_port = addressing
            .graph()
            .port_link(self.port(state, addressing, cache)?)?;
        addressing.graph().port_node(in_port)
    }

    /// The next port in the current graph if we follow one transition
    fn next_port_offset<'g, 'n, C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'n>,
        cache: &mut C,
    ) -> Option<PortOffset> {
        let in_port = addressing
            .graph()
            .port_link(self.port(state, addressing, cache)?)?;
        addressing.graph().port_offset(in_port)
    }

    /// The transitions to follow from `state`
    fn get_transitions<'g, 'n, 'm: 'n, C: AddressCache>(
        &'n self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'm>,
        cache: &mut C,
    ) -> Vec<StateTransition<Address<<Self::SpineID as SpineAddress>::AsRef<'n>>>> {
        // All transitions in `state` that are allowed for `graph`
        let out_port = self.port(state, addressing, cache);
        let in_port = out_port.and_then(|out_port| addressing.graph().port_link(out_port));
        let in_offset = in_port.map(|in_port| {
            addressing
                .graph()
                .port_offset(in_port)
                .expect("invalid port")
        });
        let next_node =
            in_port.map(|in_port| addressing.graph().port_node(in_port).expect("invalid port"));
        let mut transitions = self.trie().outputs(state).filter_map(move |out_p| {
            match self.transition(out_p, addressing) {
                StateTransition::Node(addrs, offset) => {
                    if in_offset != Some(offset) {
                        return None;
                    }
                    if addrs.iter().all(|(addressing, addr)| {
                        let Some(next_addr) = addressing.get_addr(
                                    next_node.expect("from if condition"),
                                    cache
                                ) else {
                                    return false
                                };
                        &next_addr == addr
                    }) {
                        Some(StateTransition::Node(
                            addrs.into_iter().map(|(_, addr)| addr).collect(),
                            offset,
                        ))
                    } else {
                        None
                    }
                }
                StateTransition::NoLinkedNode => {
                    // In read mode, out_port existing is enough
                    out_port.is_some().then_some(StateTransition::NoLinkedNode)
                }
                StateTransition::FAIL => Some(StateTransition::FAIL),
            }
        });
        if self.is_non_deterministic(state) {
            transitions.collect()
        } else {
            transitions.next().into_iter().collect()
        }
    }

    /// All allowed state transitions from `state`.
    fn next_states<'g, 'n, C: AddressCache>(
        &'n self,
        state: StateID,
        addressing: &Self::Addressing<'g, 'n>,
        cache: &mut C,
    ) -> Vec<StateID> {
        let addressing = if let Some(spine) = self.spine(state) {
            addressing.with_spine(spine)
        } else {
            addressing.clone()
        };
        // Compute "ideal" transition
        self.get_transitions(state, &addressing, cache)
            .into_iter()
            .map(|transition| {
                self.follow_transition(state, &transition, &addressing)
                    .expect("invalid transition")
            })
            .collect()
    }

    /// State transition from `state` following `transition`
    fn follow_transition<'g, 'n: 'm, 'm>(
        &'m self,
        state: StateID,
        transition: &StateTransition<Address<<Self::SpineID as SpineAddress>::AsRef<'m>>>,
        addressing: &Self::Addressing<'g, 'n>,
    ) -> Option<StateID> {
        self.trie()
            .outputs(state)
            .find(|&p| &self.transition(p, addressing).map(|(_, a)| a) == transition)
            .and_then(|p| {
                let in_p = self.trie().port_link(p)?;
                let n = self.trie().port_node(in_p).expect("invalid port");
                Some(n)
            })
    }
}
