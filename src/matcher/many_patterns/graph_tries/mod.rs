use std::fmt::{self, Debug, Display};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

mod base;
pub use base::BaseGraphTrie;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StateTransition<Address> {
    Node(Vec<Address>, PortOffset),
    NoLinkedNode,
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

    fn map<B, F: for<'a> FnMut(&'a A) -> B>(&self, mut f: F) -> StateTransition<B> {
        match self {
            StateTransition::Node(addrs, port) => {
                StateTransition::Node(addrs.iter().map(|addr| f(addr)).collect(), *port)
            }
            Self::NoLinkedNode => StateTransition::NoLinkedNode,
            Self::FAIL => StateTransition::FAIL,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TrieTraversal {
    ReadOnly,
    Write,
}

pub trait BoundedAddress {
    type Main: PartialEq;
    type Boundary;

    fn boundary(&self) -> &Self::Boundary;
    fn main(&self) -> &Self::Main;
}

pub trait GraphCache<Address> {
    type Boundary;

    fn get_node(&self, addr: &Address, boundary: &Self::Boundary) -> Option<NodeIndex>;
    fn get_addr(&self, node: NodeIndex, boundary: &Self::Boundary) -> Option<Address>;
    fn graph(&self) -> &PortGraph;
}

pub type StateID = NodeIndex;
pub fn root_state() -> NodeIndex {
    NodeIndex::new(0)
}

pub trait GraphTrie {
    type Address: BoundedAddress + PartialEq + Clone;

    /// The three methods to implement
    fn trie(&self) -> &PortGraph;
    fn address(&self, state: StateID) -> Option<Self::Address>;
    fn port_offset(&self, state: StateID) -> Option<PortOffset>;
    fn transition(&self, port: PortIndex) -> StateTransition<Self::Address>;
    fn is_non_deterministic(&self, state: StateID) -> bool;

    /// The node in the current graph at `state`
    fn node<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        cache: &C,
    ) -> Option<NodeIndex> {
        let addr = self.address(state)?;
        cache.get_node(&addr.main(), addr.boundary())
    }

    /// The port in the current graph at `state`
    fn port<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        cache: &C,
    ) -> Option<PortIndex> {
        let offset = self.port_offset(state)?;
        cache.graph().port_index(self.node(state, cache)?, offset)
    }

    /// The next node in the current graph if we follow one transition
    fn next_node<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        cache: &C,
    ) -> Option<NodeIndex> {
        let graph = cache.graph();
        let in_port = graph.port_link(self.port(state, cache)?)?;
        graph.port_node(in_port)
    }

    /// The next port in the current graph if we follow one transition
    fn next_port_offset<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        cache: &C,
    ) -> Option<PortOffset> {
        let graph = cache.graph();
        let in_port = graph.port_link(self.port(state, cache)?)?;
        graph.port_offset(in_port)
    }

    /// The transitions to follow from `state`
    ///
    /// Works in both read (i.e. when traversing a graph online for matching)
    /// and write (i.e. when adding patterns) modes.
    ///
    /// When reading, only return existing transitions. If the state is
    /// furthermore deterministic, then a single transition is returned.
    ///
    /// Inversely, when writing in a non-deterministic state, a single
    /// transition is returned. Furthermore, all the transitions returned
    /// will always be of a single kind of the StateTransition enum:
    ///   - StateTransition::Node(_) when there is a next node
    ///   - StateTransition::NoLinkedNode when the out_port is dangling
    ///   - StateTransition::FAIL when there is no such out_port
    /// This works because even in deterministic read-only mode, a
    /// FAIL or NoLinkedNode transition will only be followed if there is
    /// no other match
    fn get_transitions<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        graph: &PortGraph,
        partition: &C,
    ) -> Vec<StateTransition<Self::Address>> {
        // All transitions in `state` that are allowed for `graph`
        let out_port = self.port(state, partition);
        let in_port = out_port.and_then(|out_port| graph.port_link(out_port));
        let in_offset = in_port.map(|in_port| graph.port_offset(in_port).expect("invalid port"));
        let next_node = in_port.map(|in_port| graph.port_node(in_port).expect("invalid port"));
        let mut transitions = self
            .trie()
            .outputs(state)
            .filter(move |&out_p| match self.transition(out_p) {
                StateTransition::Node(ref addrs, offset) => {
                    if in_offset != Some(offset) {
                        return false;
                    }
                    addrs.iter().all(|addr| {
                        let boundary = addr.boundary();
                        let Some(next_addr) = partition.get_addr(
                                    next_node.expect("from if condition"),
                                    boundary,
                                ) else {
                                    return false
                                };
                        &next_addr == addr.main()
                    })
                }
                StateTransition::NoLinkedNode => {
                    // In read mode, out_port existing is enough
                    out_port.is_some()
                }
                StateTransition::FAIL => true,
            })
            .map(|out_p| self.transition(out_p).clone());
        if self.is_non_deterministic(state) {
            transitions.collect()
        } else {
            transitions.next().into_iter().collect()
        }
    }

    fn next_states<C: GraphCache<<Self::Address as BoundedAddress>::Main, Boundary = <Self::Address as BoundedAddress>::Boundary>>(
        &self,
        state: StateID,
        graph: &PortGraph,
        partition: &C,
    ) -> Vec<StateID> {
        // Compute "ideal" transition
        self.get_transitions(state, graph, partition)
            .into_iter()
            .map(|transition| {
                self.follow_transition(state, &transition)
                    .expect("invalid transition")
            })
            .collect()
    }

    fn follow_transition(
        &self,
        state: StateID,
        transition: &StateTransition<Self::Address>,
    ) -> Option<StateID> {
        self.trie()
            .outputs(state)
            .find(|&p| &self.transition(p) == transition)
            .and_then(|p| {
                let in_p = self.trie().port_link(p)?;
                let n = self.trie().port_node(in_p).expect("invalid port");
                Some(n)
            })
    }
}
