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

use std::{
    cmp::{self, Ordering},
    fmt::{self, Debug, Display},
};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::addressing::{
    pg::AsPathOffset, Address, AddressCache, AsSpineID, Rib, SkeletonAddressing, SpineAddress,
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
}

impl StateTransition<(Address<usize>, Vec<Rib>)> {
    /// Remove unnecessary information (ribs)
    fn into_simplified(self) -> Self {
        match self {
            StateTransition::Node(mut addrs, offset) => {
                for (addr, ribs) in addrs.iter_mut() {
                    ribs.truncate(addr.0 + 1);
                    ribs[addr.0] = [cmp::min(addr.1, 0), cmp::max(addr.1, 0)];
                }
                StateTransition::Node(addrs, offset)
            }
            StateTransition::NoLinkedNode => self,
            StateTransition::FAIL => self,
        }
    }
}

// The partial order on StateTransition is such that
// FAIL > NoLinkedNode > Node. Furthermore, within Nodes a transition is greater
// than another one if it is strictly more general, i.e.
//             node1 < node2  =>  any node1 matches also satisfies node2
impl<A: PartialEq> PartialOrd for StateTransition<(A, Vec<Rib>)> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }
        if self.transition_type() != other.transition_type() {
            return self.transition_type().partial_cmp(&other.transition_type());
        }
        let StateTransition::Node(addrs, port) = self else {
            unreachable!("FAIL and NoLinkedNode are always equal if same type")
        };
        let StateTransition::Node(other_addrs, other_port) = other else {
            unreachable!("FAIL and NoLinkedNode are always equal if same type")
        };
        if port != other_port {
            return None;
        }
        if addrs.iter().all(|(addr, ribs)| {
            other_addrs
                .iter()
                .any(|(other_addr, other_ribs)| addr == other_addr && ribs_within(ribs, other_ribs))
        }) {
            Some(Ordering::Greater)
        } else if other_addrs.iter().all(|(other_addr, other_ribs)| {
            addrs
                .iter()
                .any(|(addr, ribs)| addr == other_addr && ribs_within(other_ribs, ribs))
        }) {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

fn ribs_within(a: &[[isize; 2]], b: &[[isize; 2]]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(a, b)| a[0] >= b[0] && a[1] <= b[1])
}

/// A state in a graph trie.
///
/// Graph tries are stored themselves as port graphs, so a state is just a
/// node index in a `PortGraph`.
pub type StateID = NodeIndex;
pub(crate) fn root_state() -> NodeIndex {
    NodeIndex::new(0)
}

type GraphAddress<'n, G> = Address<<<G as GraphTrie>::SpineID as SpineAddress>::AsRef<'n>>;

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
    fn address(&self, state: StateID) -> Option<GraphAddress<'_, Self>>;

    /// The spine of a trie state.
    ///
    /// Useful for the address encoding.
    fn spine(&self, state: StateID) -> Option<&Vec<Self::SpineID>>;

    /// The port offset for the transition from `state`.
    fn port_offset(&self, state: StateID) -> Option<PortOffset>;

    /// The transition condition for the child linked at `port`.
    ///
    /// `port` must be an outgoing port of the trie.
    fn transition<'g, 'n, 'm: 'n>(
        &'n self,
        port: PortIndex,
        addressing: &Self::Addressing<'g, 'm>,
    ) -> StateTransition<(Self::Addressing<'g, 'n>, GraphAddress<'n, Self>)>;

    /// Whether the current state is not deterministic.
    fn is_non_deterministic(&self, state: StateID) -> bool;

    /// The node in the current graph at `state`
    fn node<C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'_, '_>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let addr = self.address(state)?;
        addressing.get_node(&addr, cache)
    }

    /// The port in the current graph at `state`
    fn port<C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'_, '_>,
        cache: &mut C,
    ) -> Option<PortIndex> {
        let offset = self.port_offset(state)?;
        addressing
            .graph()
            .port_index(self.node(state, addressing, cache)?, offset)
    }

    /// The next node in the current graph if we follow one transition
    fn next_node<C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'_, '_>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let in_port = addressing
            .graph()
            .port_link(self.port(state, addressing, cache)?)?;
        addressing.graph().port_node(in_port)
    }

    /// The next port in the current graph if we follow one transition
    fn next_port_offset<C: AddressCache>(
        &self,
        state: StateID,
        addressing: &Self::Addressing<'_, '_>,
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
    ) -> Vec<PortIndex> {
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
        let mut transitions = self.trie().outputs(state).filter(move |&out_p| {
            match self.transition(out_p, addressing) {
                StateTransition::Node(addrs, offset) => {
                    if in_offset != Some(offset) {
                        return false;
                    }
                    addrs.iter().all(|(addressing, addr)| {
                        let Some(next_addr) = addressing.get_addr(
                                    next_node.expect("from if condition"),
                                    cache
                                ) else {
                                    return false
                                };
                        &next_addr == addr
                    })
                }
                StateTransition::NoLinkedNode => {
                    // In read mode, out_port existing is enough
                    out_port.is_some()
                }
                StateTransition::FAIL => true,
            }
        });
        if self.is_non_deterministic(state) {
            transitions.collect()
        } else {
            transitions.next().into_iter().collect()
        }
    }

    /// All allowed state transitions from `state`.
    fn next_states<'n, C: AddressCache>(
        &'n self,
        state: StateID,
        addressing: &Self::Addressing<'_, 'n>,
        cache: &mut C,
    ) -> Vec<StateID> {
        let addressing = if let Some(spine) = self.spine(state) {
            addressing.with_spine(spine)
        } else {
            addressing.clone()
        };
        self.get_transitions(state, &addressing, cache)
            .into_iter()
            .filter_map(|out_p| {
                let in_p = self.trie().port_link(out_p)?;
                self.trie().port_node(in_p)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use portgraph::PortOffset;

    use crate::{graph_tries::StateTransition, addressing::{Address, Rib}};

    #[test]
    fn test_partial_cmp() {
        type St = StateTransition<(Address<usize>, Vec<Rib>)>;
        let o = PortOffset::new_outgoing(0);
        // the basics
        assert!(St::NoLinkedNode < St::FAIL);
        assert!(St::Node(vec![], o) < St::NoLinkedNode);
        assert!(St::Node(vec![], o) < St::FAIL);
        // more interesting
        assert!(
            St::Node(vec![((0, 2), vec![[0, 3]])], o)
                < St::Node(vec![((0, 2), vec![[0, 2]])], o)
        );
        assert!(
            St::Node(vec![((1, 2), vec![[0, 3], [-1, 2]])], o)
                < St::Node(vec![((1, 2), vec![[0, 2], [0, 2]])], o)
        );
        assert!(
            St::Node(vec![((1, 2), vec![[-1, 3], [-1, 2]])], o)
                < St::Node(vec![((1, 2), vec![[0, -1], [0, 2]])], o)
        );
        assert!(
            St::Node(vec![((1, 2), vec![[-1, 3], [-1, 2]])], o)
                == St::Node(vec![((1, 2), vec![[-1, 3], [-1, 2]])], o)
        );
    }
}