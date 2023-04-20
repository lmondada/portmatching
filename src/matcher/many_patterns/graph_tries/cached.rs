use crate::addresses::AddressWithBound;

use super::{BaseGraphTrie, BoundedAddress, GraphTrie};

pub struct CachedGraphTrie(BaseGraphTrie);

impl CachedGraphTrie {
    pub fn new(base: &BaseGraphTrie) -> Self {
        Self(base.clone())
    }
}

impl<'graph> GraphTrie<'graph> for CachedGraphTrie {
    type Address = AddressWithBound;

    fn trie(&self) -> &portgraph::PortGraph {
        self.0.trie()
    }

    fn address(&self, state: super::StateID) -> Option<Self::Address> {
        self.0.address(state)
    }

    fn port_offset(&self, state: super::StateID) -> Option<portgraph::PortOffset> {
        self.0.port_offset(state)
    }

    fn transition(&self, port: portgraph::PortIndex) -> super::StateTransition<Self::Address> {
        self.0.transition(port)
    }

    fn is_non_deterministic(&self, state: super::StateID) -> bool {
        self.0.is_non_deterministic(state)
    }

    fn node<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        cache: &C,
    ) -> Option<portgraph::NodeIndex> {
        let addr = self.address(state)?;
        cache.get_node(&addr.main(), addr.boundary())
    }

    fn port<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        cache: &C,
    ) -> Option<portgraph::PortIndex> {
        let offset = self.port_offset(state)?;
        cache.graph().port_index(self.node(state, cache)?, offset)
    }

    fn next_node<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        cache: &C,
    ) -> Option<portgraph::NodeIndex> {
        let graph = cache.graph();
        let in_port = graph.port_link(self.port(state, cache)?)?;
        graph.port_node(in_port)
    }

    fn next_port_offset<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        cache: &C,
    ) -> Option<portgraph::PortOffset> {
        let graph = cache.graph();
        let in_port = graph.port_link(self.port(state, cache)?)?;
        graph.port_offset(in_port)
    }

    fn get_transitions<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        graph: &'graph portgraph::PortGraph,
        partition: &C,
    ) -> Vec<super::StateTransition<Self::Address>> {
        // All transitions in `state` that are allowed for `graph`
        let out_port = self.port(state, partition);
        let in_port = out_port.and_then(|out_port| graph.port_link(out_port));
        let in_offset = in_port.map(|in_port| graph.port_offset(in_port).expect("invalid port"));
        let next_node = in_port.map(|in_port| graph.port_node(in_port).expect("invalid port"));
        let mut transitions = self
            .trie()
            .outputs(state)
            .filter(move |&out_p| match self.transition(out_p) {
                super::StateTransition::Node(ref addrs, offset) => {
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
                super::StateTransition::NoLinkedNode => {
                    // In read mode, out_port existing is enough
                    out_port.is_some()
                }
                super::StateTransition::FAIL => true,
            })
            .map(|out_p| self.transition(out_p).clone());
        if self.is_non_deterministic(state) {
            transitions.collect()
        } else {
            transitions.next().into_iter().collect()
        }
    }

    fn next_states<C: super::GraphCache<'graph, Self::Address>>(
        &self,
        state: super::StateID,
        graph: &'graph portgraph::PortGraph,
        partition: &C,
    ) -> Vec<super::StateID> {
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
        state: super::StateID,
        transition: &super::StateTransition<Self::Address>,
    ) -> Option<super::StateID> {
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
