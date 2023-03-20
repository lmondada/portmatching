use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::{self, Debug, Display},
    fs,
};

use bimap::BiBTreeMap;
use portgraph::{
    algorithms::postorder, dot::dot_string_weighted, Direction, NodeIndex, PortGraph, PortIndex,
    SecondaryMap, Weights,
};

use crate::{pattern::Edge, utils::cover::cover_nodes, PortOffset};

use super::{ManyPatternMatcher, ReadGraphTrie, WriteGraphTrie};

/// Every node in a pattern has an address, given by the line index and
/// the index within the line
///
/// An address is unique within a pattern, and will always be unique within
/// a maximum match as well
///
/// Note that a same node in a graph may have more than one address. In this
/// case, we always refer to the node by its smallest address, so that the
/// map address <-> node ID is bijective
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternNodeAddress(usize, usize);

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PermPortIndex(usize);

impl PermPortIndex {
    fn next(&self) -> Self {
        let PermPortIndex(ind) = self;
        PermPortIndex(ind + 1)
    }
}

impl PatternNodeAddress {
    /// The root of the pattern is always assigned the address (0, 0)
    const ROOT: PatternNodeAddress = Self(0, 0);

    /// The next address on the same line
    fn next(&self) -> Self {
        Self(self.0, self.1 + 1)
    }

    /// The first (root) address on the next line
    fn next_root(&self) -> Self {
        Self(self.0 + 1, 0)
    }
}

impl Debug for PatternNodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", (self.0, self.1))
    }
}

/// All possible transitions in the Graph trie
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum NodeTransition {
    /// Following the current edge leads to a known node, with the given addr
    /// and coming in through `port`
    KnownNode(PatternNodeAddress, PortOffset),
    /// Following the current edge leads to an unknown node, and coming in
    /// through `port`
    NewNode(PortOffset),
    /// The current edge is not linked to any other vertex
    NoLinkedNode,
    /// The current edge does not exist. For root vertices in the trie,
    /// fail transitions also indicate an epsilon transition
    Fail,
}

impl fmt::Display for NodeTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeTransition::KnownNode(PatternNodeAddress(i, j), port) => {
                write!(f, "({}, {})[{}]", i, j, port)
            }
            NodeTransition::NewNode(port) => {
                write!(f, "X[{}]", port)
            }
            NodeTransition::NoLinkedNode => write!(f, "dangling"),
            NodeTransition::Fail => write!(f, "fail"),
        }
    }
}

/// A node in the NaiveGraphTrie
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `find_port` serves to store the map NodeTransition => PortIndex.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, Default, Debug)]
struct NodeWeight {
    port_offset: Option<PortOffset>,
    address: Option<PatternNodeAddress>,
    transitions: BTreeMap<NodeTransition, PortIndex>,
    non_deterministic: bool,
}
impl NodeWeight {
    fn get_port(&self, transition: &NodeTransition) -> Option<PortIndex> {
        self.transitions.get(transition).copied()
    }
}

/// A simple implementation of a graph trie
///
/// The graph trie is organised in "lines". Within each line (corresponding
/// to a maximal line of the patterns), the node transitions are deterministic.
/// Between lines, transitions are non-deterministic, so that more than one
/// line may have to be traversed concurrently at any one time.
pub struct NaiveGraphTrie {
    graph: PortGraph,
    weights: SecondaryMap<NodeIndex, NodeWeight>,
    perm_indices: RefCell<BTreeMap<PermPortIndex, PortIndex>>,
}

impl Default for NaiveGraphTrie {
    fn default() -> Self {
        let graph = PortGraph::default();
        let weights = SecondaryMap::default();
        let perm_indices = Default::default();
        let mut ret = Self {
            graph,
            weights,
            perm_indices,
        };
        ret.add_node(true);
        ret
    }
}

#[derive(PartialEq, Eq)]
enum TrieTraversal {
    ReadOnly,
    Write,
}

impl NaiveGraphTrie {
    /// An empty graph trie
    pub fn new() -> Self {
        Self::default()
    }

    /// Follow a transition from `node` along `transition`
    ///
    /// Returns None if the transition does not exist
    fn transition(&self, node: NodeIndex, transition: &NodeTransition) -> Option<NodeIndex> {
        let out_port = self.weights[node].get_port(transition)?;
        let in_port = self.graph.port_link(out_port)?;
        self.graph.port_node(in_port)
    }

    fn transition_port(
        &self,
        node: NodeIndex,
        transition: &NodeTransition,
    ) -> Option<PermPortIndex> {
        let out_port = self.weights[node].get_port(transition)?;
        let in_port = self.graph.port_link(out_port)?;
        self.get_perm_port_index(in_port).into()
    }

    fn transitions(&self, node: NodeIndex) -> impl Iterator<Item = (&NodeTransition, NodeIndex)> {
        self.weights[node]
            .transitions
            .iter()
            .filter_map(|(transition, &out_port)| {
                let in_port = self.graph.port_link(out_port)?;
                self.graph.port_node(in_port).map(|node| (transition, node))
            })
    }

    fn transitions_port(
        &self,
        node: NodeIndex,
    ) -> impl Iterator<Item = (&NodeTransition, PermPortIndex)> {
        self.weights[node]
            .transitions
            .iter()
            .filter_map(|(transition, &out_port)| {
                let in_port = self.graph.port_link(out_port)?;
                (transition, self.get_perm_port_index(in_port)).into()
            })
    }

    /// Set a new transition between `node` and `next_node`
    ///
    /// If the transition already exists, then trying to update it to a
    /// different value will return None. In most cases this should be
    /// considered a logic mistake.
    #[must_use]
    fn set_transition(
        &mut self,
        node: NodeIndex,
        next_node: NodeIndex,
        transition: NodeTransition,
    ) -> Option<PortIndex> {
        // Cannot use .entry because of borrow checker
        let out_port = self.weights[node]
            .transitions
            .get(&transition)
            .copied()
            .unwrap_or_else(|| {
                self.add_port(node, Direction::Outgoing);
                let out_port = self.graph.outputs(node).last().expect("Just added");
                self.weights[node].transitions.insert(transition, out_port);
                out_port
            });
        let in_port = self.graph.port_link(out_port).unwrap_or_else(|| {
            self.add_port(next_node, Direction::Incoming);
            let in_port = self.graph.inputs(next_node).last().expect("Just added");
            self.graph.link_ports(out_port, in_port).unwrap();
            in_port
        });
        (self.graph.port_node(in_port).expect("Invalid port") == next_node).then_some(in_port)
    }

    fn get_perm_port_index(&self, port: PortIndex) -> PermPortIndex {
        let next_ind = self
            .perm_indices
            .borrow()
            .keys()
            .max()
            .map(|p| p.next())
            .unwrap_or_default();
        self.perm_indices.borrow_mut().insert(next_ind, port);
        next_ind
    }

    fn get_state(&self, port: PermPortIndex) -> Option<NodeIndex> {
        let &port = self.perm_indices.borrow().get(&port)?;
        self.graph.port_node(port)
    }

    fn free(&self, port: PermPortIndex) -> Option<PortIndex> {
        self.perm_indices.borrow_mut().remove(&port)
    }

    fn free_node(&self, port: PermPortIndex) -> Option<NodeIndex> {
        let port = self.perm_indices.borrow_mut().remove(&port)?;
        self.graph.port_node(port)
    }

    fn add_port(&mut self, node: NodeIndex, dir: Direction) {
        let (incoming, outgoing) = (self.graph.num_inputs(node), self.graph.num_outputs(node));
        let (incoming, outgoing) = match dir {
            Direction::Incoming => (incoming + 1, outgoing),
            Direction::Outgoing => (incoming, outgoing + 1),
        };
        self.set_num_ports(node, incoming, outgoing);
    }

    fn set_num_ports(&mut self, node: NodeIndex, incoming: usize, outgoing: usize) {
        self.graph
            .set_num_ports(node, incoming, outgoing, |old, new, graph| {
                let node = graph.port_node(old).expect("Invalid port");
                for val in self.weights[node].transitions.values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
                for val in self.perm_indices.borrow_mut().values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
            });
    }

    // /// The node at address `node`
    fn state(&self, node: NodeIndex) -> &NodeWeight {
        &self.weights[node]
    }

    // /// A mutable reference to the `node` state
    fn state_mut(&mut self, node: NodeIndex) -> &mut NodeWeight {
        &mut self.weights[node]
    }

    /// Extend an existing tree to handle a longer maximal line
    ///
    /// This is in principle similar to `append_tree`, except that it preserves
    /// all the "business logic" of the graph trie.
    /// That is to say, it will do a lot of nodes copying so that any patterns
    /// that were previously added to the trie will still match along the newly
    /// added extension
    ///
    /// This is to be refactored and cleaned up -- there is a lot going on here.
    fn extend_tree(
        &mut self,
        node: NodeIndex,
        addr: PatternNodeAddress,
        new_node: &mut Option<NodeIndex>,
    ) -> NodeIndex {
        let new_node = *new_node.get_or_insert_with(|| self.add_node(false));
        if self.weights[new_node].address.as_ref() < Some(&addr) {
            self.weights[new_node].address = addr.clone().into();
        }
        let fallback = if self.weights[node].non_deterministic {
            self.transition(node, &NodeTransition::NoLinkedNode)
        } else {
            self.transition(node, &NodeTransition::NoLinkedNode)
                .or_else(|| self.transition(node, &NodeTransition::Fail))
        };
        if let Some(fallback) = fallback {
            // For each NewNode(port) transition in fallback, we need to add the
            // KnownNode(addr, port) transition for the new addr
            let descendants: Vec<_> = self.all_nodes_in(fallback).collect();
            for node in descendants {
                let new_node_transitions: Vec<_> = self
                    .transitions(node)
                    .filter_map(|(label, next_node)| {
                        if let NodeTransition::NewNode(port) = label {
                            Some((port.clone(), next_node))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (port, next_node) in new_node_transitions {
                    self.set_transition(
                        node,
                        next_node,
                        NodeTransition::KnownNode(addr.clone(), port.clone()),
                    )
                    .unwrap();
                }
            }
            self.set_transition(new_node, fallback, NodeTransition::Fail)
                .expect("Changing existing transition");
        }
        new_node
    }

    /// Whether coordinates are at the beginning of a tree
    fn is_root(&self, root: NodeIndex) -> bool {
        self.weights[root].non_deterministic
    }

    /// A vector of all nodes that are descendants of `root`
    ///
    /// We return them in post order, but the order does not matter
    fn all_nodes_in(&self, root: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        postorder(&self.graph, [root], Direction::Outgoing)
    }

    fn get_transition(
        &self,
        state: NodeIndex,
        graph: &PortGraph,
        current_match: &MatchObject,
        traversal: TrieTraversal,
    ) -> Vec<(NodeTransition, MatchObject)> {
        // The transition we would want to perform
        let ideal_transition = || {
            let addr = self.state(state).address.as_ref()?;
            let graph_node = *current_match
                .map
                .get_by_right(addr)
                .expect("Malformed pattern trie");
            let out_port = self.state(state).port_offset.clone()?;
            let out_port = out_port.get_index(graph_node, graph)?;
            let in_port = graph.port_link(out_port);

            Some(match in_port {
                Some(in_port) => {
                    let port_offset = PortOffset::try_from_index(in_port, graph).unwrap();
                    let in_node = graph.port_node(in_port).unwrap();
                    match current_match.map.get_by_left(&in_node).cloned() {
                        Some(new_addr) => NodeTransition::KnownNode(new_addr, port_offset),
                        None => NodeTransition::NewNode(port_offset),
                    }
                }
                None => NodeTransition::NoLinkedNode,
            })
        };
        let ideal_transition = ideal_transition().unwrap_or(NodeTransition::Fail);

        // From the ideal transition, compute all valid transitions
        // (depending on whether the trie traversal is read only or write)
        if traversal == TrieTraversal::ReadOnly {
            // Fall-back to simpler transitions if non-existent
            let transition = self
                .fallback_transition(state, &ideal_transition)
                .unwrap_or(NodeTransition::Fail);
            let next_match = self.compute_next_match(current_match, state, graph, &transition);
            vec![(transition, next_match)]
        } else {
            let next_match =
                self.compute_next_match(current_match, state, graph, &ideal_transition);
            let mut all_transitions = Vec::new();
            if let NodeTransition::NewNode(port) = &ideal_transition {
                all_transitions.extend(current_match.no_map.iter().map(|addr| {
                    (
                        NodeTransition::KnownNode(addr.clone(), port.clone()),
                        next_match.clone(),
                    )
                }));
            }
            all_transitions.push((ideal_transition, next_match));
            all_transitions
        }
    }

    fn compute_next_match(
        &self,
        current_match: &MatchObject,
        state: NodeIndex,
        graph: &PortGraph,
        transition: &NodeTransition,
    ) -> MatchObject {
        let mut next_match = current_match.clone();
        let in_port = match &transition {
            NodeTransition::Fail => {
                if !self.is_root(state) {
                    let next_addr = current_match.current_addr.next_root();
                    next_match.current_addr = next_addr;
                }
                return next_match;
            }
            NodeTransition::NoLinkedNode => {
                let next_addr = current_match.current_addr.next_root();
                next_match.current_addr = next_addr;
                return next_match;
            }
            NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                let addr = &self.state(state).address.as_ref().unwrap();
                let graph_node = *current_match
                    .map
                    .get_by_right(addr)
                    .expect("Malformed pattern trie");
                let out_port = self.state(state).port_offset.clone().unwrap();
                let out_port = out_port.get_index(graph_node, graph).unwrap();
                next_match.current_addr = current_match.current_addr.next();
                graph.port_link(out_port).unwrap()
            }
        };

        let next_graph_node = graph.port_node(in_port).expect("Invalid port");
        let next_addr = match &transition {
            NodeTransition::KnownNode(addr, _) => {
                if let Some(next_state) = self.transition(state, &transition) {
                    if let Some(next_addr) = &self.weights[next_state].address {
                        // overwrite the addr to next_addr
                        next_match.map.remove_by_right(addr);
                        next_addr.clone()
                    } else {
                        addr.clone()
                    }
                } else {
                    addr.clone()
                }
            }
            NodeTransition::NewNode(_) => next_match.current_addr.clone(),
            NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                unreachable!()
            }
        };

        // If we are encountering a known node, but that node was only known
        // in no map, we need to remove it from that map before inserting it
        // into the match map
        next_match.no_map.remove(&next_addr);
        next_match
            .map
            .insert_no_overwrite(next_graph_node, next_addr.clone())
            .ok()
            .or_else(|| {
                (next_match.map.get_by_left(&next_graph_node) == Some(&next_addr)).then_some(())
            })
            .expect("Map is not injective");
        next_match
    }

    /// Assign `node` to given address and port, if possible
    ///
    /// Will update the node fields to the given values if those are None.
    /// Returns whether it was successful
    fn try_update_node(
        &mut self,
        node: NodeIndex,
        addr: &PatternNodeAddress,
        port_index: &PortOffset,
    ) -> bool {
        let node = self.state_mut(node);
        let port_offset = node.port_offset.as_ref().unwrap_or(port_index);
        let address = node.address.as_ref().unwrap_or(addr);
        if port_offset == port_index && address == addr {
            node.port_offset = port_offset.clone().into();
            node.address = address.clone().into();
            true
        } else {
            false
        }
    }

    /// Adds Fail or NoLinkedNode transition at `state`
    ///
    /// If `is_dangling`, the transition is a NoLinkedNode, otherwise Fail
    fn add_default_transition(
        &mut self,
        state: NodeIndex,
        is_dangling: bool,
        new_node: &mut Option<NodeIndex>,
    ) {
        let transition = if is_dangling {
            NodeTransition::NoLinkedNode
        } else {
            NodeTransition::Fail
        };
        if self.transition(state, &transition).is_none() {
            let next_node = if transition == NodeTransition::NoLinkedNode && !self.is_root(state) {
                self.transition(state, &NodeTransition::Fail)
                    .unwrap_or_else(|| *new_node.get_or_insert_with(|| self.add_node(true)))
            } else {
                *new_node.get_or_insert_with(|| self.add_node(true))
            };
            self.set_transition(state, next_node, transition).unwrap();
        }
    }

    /// Finds all possible new lines to transition to
    ///
    /// When inserting "lines" from a pattern into the graph trie, at every
    /// new line one must find all possible transitions that lead to new line;
    /// the "carriage return" operation.
    fn carriage_return(
        &mut self,
        state: NodeIndex,
        current_match: MatchObject,
        new_node: &mut Option<NodeIndex>,
        is_dangling: bool,
    ) -> Vec<(PermPortIndex, MatchObject)> {
        let mut current_states = vec![(state, current_match)];
        let mut next_states = Vec::new();
        assert!(!self.is_root(state) || is_dangling);

        while let Some((current_state, current_match)) = current_states.pop() {
            // Insert new Fail/NoSuchLink transition if it does not exist
            self.add_default_transition(
                current_state,
                is_dangling && current_state == state,
                new_node,
            );
            for (transition, next_port) in self.transitions_port(current_state) {
                match transition {
                    NodeTransition::KnownNode(_, _) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        current_states
                            .push((self.free_node(next_port).expect("Invalid port"), next_match));
                    }
                    NodeTransition::NewNode(_) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr.clone();
                        next_match.no_map.insert(next_addr);
                        current_states
                            .push((self.free_node(next_port).expect("Invalid port"), next_match));
                    }
                    NodeTransition::NoLinkedNode => {
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((next_port, next_match));
                    }
                    NodeTransition::Fail => {
                        if is_dangling && current_state == state {
                            continue;
                        }
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((next_port, next_match));
                    }
                }
            }
        }
        next_states
    }

    /// Follow Fail transitions to find state with `out_port`
    fn valid_start_states(
        &mut self,
        states: Vec<(NodeIndex, MatchObject)>,
        out_port: portgraph::PortIndex,
        graph: &PortGraph,
    ) -> (Vec<(NodeIndex, MatchObject)>, Vec<PermPortIndex>) {
        // We start by finding where we are in the graph, using the outgoing
        // port
        let graph_node = graph.port_node(out_port).expect("Invalid port");
        let graph_port = PortOffset::try_from_index(out_port, graph).expect("Invalid port");

        // Process carriage returns where necessary
        let mut states_post_carriage_return = Vec::new();
        // All in-ports that are traversed - useful to split graph into owned
        // states
        let mut all_ports = Vec::new();

        // If the current state does not correspond to our position and we
        // are in a deterministic state, follow the FAIL transition (aka do a
        // carriage return)
        let mut new_fail_state = None;
        for (state, current_match) in states {
            let graph_addr = current_match
                .map
                .get_by_left(&graph_node)
                .expect("Incomplete match map");
            if !self.try_update_node(state, graph_addr, &graph_port) && !self.is_root(state) {
                let (mut ports, matches): (Vec<_>, Vec<_>) = self
                    .carriage_return(state, current_match.clone(), &mut new_fail_state, false)
                    .into_iter()
                    .unzip();
                states_post_carriage_return.extend(
                    ports
                        .iter()
                        .zip(matches)
                        .map(|(p, m)| (self.get_state(*p).expect("Invalid port"), m)),
                );
                all_ports.append(&mut ports);
            } else {
                states_post_carriage_return.push((state, current_match));
            }
        }

        // Finally we obtain the states where we are meant to be,
        // such that state_addr == edge_addr and state_port == edge_port
        let mut new_node = None;
        let states = states_post_carriage_return
            .into_iter()
            .map(|(mut state, current_match)| {
                let graph_addr = current_match
                    .map
                    .get_by_left(&graph_node)
                    .expect("Incomplete match map");
                while !self.try_update_node(state, graph_addr, &graph_port) {
                    let port = match self.transition_port(state, &NodeTransition::Fail) {
                        Some(next_port) => next_port,
                        None => {
                            let new_node = *new_node.get_or_insert_with(|| self.add_node(true));
                            let port = self
                                .set_transition(state, new_node, NodeTransition::Fail)
                                .expect("Changing existing value");
                            self.get_perm_port_index(port)
                            // new_node
                        }
                    };
                    all_ports.push(port);
                    state = self.get_state(port).expect("Invalid port");
                    assert!(self.is_root(state));
                }
                (state, current_match)
            })
            .collect();
        (states, all_ports)
    }

    fn add_node(&mut self, non_deterministic: bool) -> NodeIndex {
        let node = self.graph.add_node(0, 0);
        self.weights[node].non_deterministic = non_deterministic;
        node
    }

    /// Output trie graph in dotstring format
    pub fn dotstring(&self) -> String {
        self.dotstring_with(&BTreeMap::<_, Vec<String>>::new())
    }

    /// Output trie graph in dotstring format
    pub fn dotstring_with<D: Display>(
        &self,
        matching_nodes: &BTreeMap<NodeIndex, Vec<D>>,
    ) -> String {
        let mut node_weights = SecondaryMap::new();
        let mut port_weights = SecondaryMap::new();
        for node in self.graph.nodes_iter() {
            let mut fmt_str = String::new();
            if let Some(addr) = &self.weights[node].address {
                fmt_str += &format!("({}, {})", addr.0, addr.1);
            } else {
                fmt_str += "None";
            }
            if let Some(port) = &self.weights[node].port_offset {
                fmt_str += &format!("[{}]", port);
            }
            if let Some(matches) = matching_nodes.get(&node) {
                if !matches.is_empty() {
                    fmt_str += " [[";
                    fmt_str += &matches
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    fmt_str += "]]";
                }
            }
            node_weights[node] = fmt_str;

            for port in self.graph.outputs(node) {
                if let Some((transition, _)) = self.weights[node]
                    .transitions
                    .iter()
                    .find(|(_, p)| port == **p)
                {
                    port_weights[port] = transition.to_string();
                }
            }
        }
        let weights = Weights {
            nodes: node_weights,
            ports: port_weights,
        };
        dot_string_weighted(&self.graph, &weights)
    }

    /// Split states so that all incoming links are within ports
    ///
    /// Ports are all incoming ports -- split states so that a state
    /// is incident to one of the port iff all ports are within `ports`
    fn to_owned_states<F, I1, I2>(&mut self, prev_states: I1, all_ports: I2, mut clone_state: F)
    where
        F: FnMut(NodeIndex, NodeIndex),
        I1: IntoIterator<Item = NodeIndex>,
        I2: IntoIterator<Item = PermPortIndex>,
    {
        let cover = prev_states.into_iter().collect();
        let all_ports = all_ports
            .into_iter()
            .map(|p| self.free(p).expect("Invalid port"))
            .collect();
        let weights = RefCell::new(&mut self.weights);
        cover_nodes(
            &mut self.graph,
            &cover,
            all_ports,
            |state, new_state, graph| {
                let mut weights = weights.borrow_mut();
                weights[new_state] = weights[state].clone();
                // update transition pointers
                let [old_weight, new_weight] =
                    weights.get_disjoint_mut([state, new_state]).unwrap();
                for (&out_port, new_out_port) in old_weight
                    .transitions
                    .values()
                    .zip(new_weight.transitions.values_mut())
                {
                    let offset = graph.port_offset(out_port).expect("Invalid port");
                    *new_out_port = graph
                        .port_index(new_state, offset, Direction::Outgoing)
                        .unwrap();
                }
                // callback
                clone_state(state, new_state);
            },
            |old, new, graph| {
                let node = graph.port_node(old).expect("Invalid port");
                for val in weights.borrow_mut()[node].transitions.values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
                for val in self.perm_indices.borrow_mut().values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
            },
        );
    }

    /// Finds the first existing transition that is compatible with `transition`
    fn fallback_transition(
        &self,
        state: NodeIndex,
        transition: &NodeTransition,
    ) -> Option<NodeTransition> {
        let mut transition = transition.clone();
        while self.transition(state, &transition).is_none() {
            transition = match transition {
                NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                    NodeTransition::NoLinkedNode
                }
                NodeTransition::NoLinkedNode => NodeTransition::Fail,
                NodeTransition::Fail => return None,
            };
        }
        transition.into()
    }
}

// fn rekey<'w>(
//     graph: &PortGraph,
//     weights: &'w RefCell<&'w mut SecondaryMap<NodeIndex, NodeWeight>>,
// ) -> impl FnMut(PortIndex, Option<PortIndex>) + 'w {
//     let port2node: BTreeMap<_, _> = graph
//         .ports_iter()
//         .map(|p| (p, graph.port_node(p).expect("Invalid port")))
//         .collect();
//     move |old, new| {
//         let node = port2node[&old];
//         for val in weights.borrow_mut()[node].transitions.values_mut() {
//             if &old == val {
//                 *val = new.expect("A linked port was deleted!");
//             }
//         }
//     }
// }

/// An object to store a pattern match during matching
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MatchObject {
    map: BiBTreeMap<NodeIndex, PatternNodeAddress>,
    // Entries in `no_map` are in `map` but are irrelevant
    no_map: BTreeSet<PatternNodeAddress>,
    current_addr: PatternNodeAddress,
}

impl MatchObject {
    fn new(left_root: NodeIndex) -> Self {
        Self {
            map: BiBTreeMap::from_iter([(left_root, PatternNodeAddress::ROOT)]),
            no_map: BTreeSet::new(),
            current_addr: PatternNodeAddress::ROOT,
        }
    }
}

impl WriteGraphTrie for NaiveGraphTrie {
    fn create_next_states<F: FnMut(Self::StateID, Self::StateID)>(
        &mut self,
        states: Vec<(Self::StateID, Self::MatchObject)>,
        &Edge(out_port, _): &Edge,
        graph: &PortGraph,
        clone_state: F,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        // Find the states that correspond to out_port
        let prev_states: Vec<_> = states.iter().map(|(state, _)| state.clone()).collect();
        let (states, mut all_ports) = self.valid_start_states(states, out_port, graph);

        let mut next_states = BTreeSet::new();
        let mut new_node = None;
        for (state, current_match) in states {
            // For every allowable transition, insert it if it does not exist
            // and return it
            let (mut next_ports, next_matches): (Vec<_>, Vec<_>) = self
                .get_transition(state, graph, &current_match, TrieTraversal::Write)
                .into_iter()
                .flat_map(|(transition, next_match)| {
                    if let NodeTransition::NoLinkedNode = &transition {
                        self.carriage_return(state, current_match.clone(), &mut new_node, true)
                    } else {
                        [(
                            self.transition_port(state, &transition).unwrap_or_else(|| {
                                let next_addr = match &transition {
                                    NodeTransition::KnownNode(addr, _) => addr.clone(),
                                    NodeTransition::NewNode(_) => next_match.current_addr.clone(),
                                    NodeTransition::NoLinkedNode => unreachable!(),
                                    NodeTransition::Fail => {
                                        panic!("transition is not valid")
                                    }
                                };
                                let next_state = self.extend_tree(state, next_addr, &mut new_node);
                                let port = self
                                    .set_transition(state, next_state, transition)
                                    .expect("Changing existing value");
                                self.get_perm_port_index(port)
                            }),
                            next_match,
                        )]
                        .into()
                    }
                })
                .unzip();
            next_states.extend(
                next_ports
                    .iter()
                    .map(|&p| self.get_state(p).expect("Invalid port"))
                    .zip(next_matches),
            );
            all_ports.append(&mut next_ports);
        }

        // Finally, wherever not all inputs come from known nodes, split the
        // state into two
        self.to_owned_states(prev_states, all_ports, clone_state);
        next_states.into_iter().collect()
    }
}

impl ReadGraphTrie for NaiveGraphTrie {
    type StateID = NodeIndex;

    type MatchObject = MatchObject;

    fn init(&self, root: NodeIndex) -> (Self::StateID, Self::MatchObject) {
        (NodeIndex::new(0), MatchObject::new(root))
    }

    fn next_states(
        &self,
        state: &Self::StateID,
        graph: &PortGraph,
        current_match: &Self::MatchObject,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        // Compute "ideal" transition
        let mut next_states: Vec<_> = self
            .get_transition(*state, graph, current_match, TrieTraversal::ReadOnly)
            .into_iter()
            .filter_map(|(transition, current_match)| {
                Some((self.transition(*state, &transition)?, current_match))
            })
            .collect();

        // Add epsilon transition if we are at a root (i.e. non-deterministic)
        if self.is_root(*state) {
            if let Some(next_state) = self.transition(*state, &NodeTransition::Fail) {
                next_states.push((next_state, current_match.clone()));
            }
        }
        next_states
    }
}

pub type NaiveManyPatternMatcher = ManyPatternMatcher<NaiveGraphTrie>;

#[cfg(test)]
mod tests {
    use std::fs;

    #[cfg(feature = "serde")]
    use itertools::Itertools;
    use portgraph::{dot::dot_string, proptest::gen_portgraph, Direction, NodeIndex, PortGraph};

    use proptest::prelude::*;

    use crate::{
        matcher::{
            many_patterns::{naive::NaiveManyPatternMatcher, PatternID, PatternMatch},
            Matcher, SinglePatternMatcher,
        },
        pattern::Pattern,
        utils::test_utils::gen_portgraph_connected,
    };

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, 0, Direction::Outgoing).unwrap(),
            g.port_index(n, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                id: PatternID(0),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_loop_link2() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, 0, Direction::Outgoing).unwrap(),
            g.port_index(n, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                id: PatternID(0),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, 0, Direction::Outgoing).unwrap(),
            g.port_index(n1, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    #[test]
    fn single_pattern_construction() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(3, 2);
        let n1 = g.add_node(2, 0);
        let n2 = g.add_node(2, 1);
        link(&mut g, (n0, 0), (n0, 2));
        link(&mut g, (n0, 1), (n1, 1));
        link(&mut g, (n2, 0), (n0, 1));
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);
    }

    fn link(p: &mut PortGraph, (n1, p1): (NodeIndex, usize), (n2, p2): (NodeIndex, usize)) {
        p.link_ports(
            p.port_index(n1, p1, Direction::Outgoing).unwrap(),
            p.port_index(n2, p2, Direction::Incoming).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn two_simple_patterns() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(0, 1);
        let n1 = p1.add_node(2, 0);
        let n2 = p1.add_node(0, 1);
        link(&mut p1, (n0, 0), (n1, 0));
        link(&mut p1, (n2, 0), (n1, 1));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 2);
        let n1 = g.add_node(3, 1);
        let n2 = g.add_node(0, 2);
        let n3 = g.add_node(3, 2);
        link(&mut g, (n0, 0), (n1, 2));
        link(&mut g, (n0, 1), (n3, 0));
        link(&mut g, (n1, 0), (n3, 1));
        link(&mut g, (n2, 0), (n1, 0));
        link(&mut g, (n2, 1), (n3, 2));
        link(&mut g, (n3, 0), (n1, 1));
        link(&mut g, (n3, 1), (n0, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
        assert_eq!(matcher.find_matches(&g).len(), 3);
    }

    #[test]
    fn trie_construction_fail() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(2, 1);
        let n1 = p1.add_node(1, 0);
        link(&mut p1, (n0, 0), (n1, 0));
        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 0);
        let n1 = p2.add_node(0, 2);
        link(&mut p2, (n1, 0), (n0, 1));
        link(&mut p2, (n1, 1), (n0, 0));
        NaiveManyPatternMatcher::from_patterns(
            [p1, p2].map(|p| Pattern::from_graph(p).unwrap()).into(),
        );
    }

    #[test]
    fn two_simple_patterns_2() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(1, 0);
        let n1 = p1.add_node(1, 3);
        let n2 = p1.add_node(0, 3);
        let n3 = p1.add_node(1, 3);
        link(&mut p1, (n1, 0), (n0, 0));
        link(&mut p1, (n2, 0), (n1, 0));
        link(&mut p1, (n2, 2), (n3, 0));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
        assert_eq!(matcher.find_matches(&g).len(), 1);
    }
    #[test]
    fn two_simple_patterns_3() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(3, 1);
        let n1 = p1.add_node(0, 3);
        link(&mut p1, (n1, 1), (n0, 0));
        link(&mut p1, (n0, 0), (n0, 2));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(0, 2);
        let n1 = p2.add_node(2, 0);
        link(&mut p2, (n0, 0), (n1, 1));
        link(&mut p2, (n0, 1), (n1, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_graph(pattern).unwrap();
            let mut matcher = NaiveManyPatternMatcher::new();
            let pattern_id = matcher.add_pattern(pattern.clone());
            let single_matcher = SinglePatternMatcher::from_pattern(pattern.clone());
            let many_matches = matcher.find_matches(&g);
            let single_matches: Vec<_> = single_matcher
                .find_matches(&g)
                .into_iter()
                .map(|m| PatternMatch {
                    id: pattern_id,
                    root: m[&pattern.root],
                })
                .collect();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "very slow"]
        #[cfg(feature = "serde")]
        #[test]
        fn many_graphs_proptest(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..4),
            g in gen_portgraph(30, 4, 60)
        ) {
            for (i, p) in patterns.iter().enumerate() {
                fs::write(&format!("pattern_{}.bin", i), rmp_serde::to_vec(p).unwrap()).unwrap();
            }
            fs::write("graph.bin", rmp_serde::to_vec(&g).unwrap()).unwrap();
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            let single_matchers = patterns
                .clone()
                .into_iter()
                .map(SinglePatternMatcher::from_pattern)
                .collect_vec();
            let single_matches = single_matchers
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    m.find_matches(&g)
                        .into_iter()
                        .map(|m| PatternMatch {
                            id: PatternID(i),
                            root: m[&patterns[i].root],
                        })
                        .collect_vec()
                })
                .collect_vec();
            fs::write("results.bin", rmp_serde::to_vec(&single_matches).unwrap()).unwrap();
            let matcher = NaiveManyPatternMatcher::from_patterns(patterns.clone());
            let many_matches = matcher.find_matches(&g);
            let many_matches = (0..patterns.len())
                .map(|i| {
                    many_matches
                        .iter()
                        .filter(|m| m.id == PatternID(i))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    // #[test]
    // fn traverse_from_test() {
    //     let mut matcher = ManyPatternMatcher::new();
    //     let tree = &mut matcher.tree.line_trees[0];
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(0)),
    //         TreeNodeID::SameTree(1),
    //     );
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(1)),
    //         TreeNodeID::SameTree(2),
    //     );
    //     tree.push(TreeNode {
    //         out_port: Some(PortOffset(1)),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(
    //             NodeTransition::NewNode(PortOffset(0)),
    //             TreeNodeID::SameTree(3),
    //         )]
    //         .into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(2).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(1).into(),
    //         address: PatternNodeAddress(0, 2),
    //         transitions: [].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(0).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v3 = g.add_node(1, 1);
    //     let v4 = g.add_node(1, 0);
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v1_0 = g.port_index(v1, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v0_0, v1_0).unwrap();
    //     let v1_1 = g.port_index(v1, 0, Direction::Outgoing).unwrap();
    //     let v3_0 = g.port_index(v3, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v1_1, v3_0).unwrap();
    //     let v3_1 = g.port_index(v3, 0, Direction::Outgoing).unwrap();
    //     let v4_0 = g.port_index(v4, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v3_1, v4_0).unwrap();
    //     let line = vec![
    //         Edge(v0_0, v1_0.into()),
    //         Edge(v1_1, v3_0.into()),
    //         Edge(v3_1, v4_0.into()),
    //     ];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line),
    //         (3, Some(&Edge(v3_1, v4_0.into())))
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v1, PatternNodeAddress(0, 1)),
    //             (v3, PatternNodeAddress(0, 2)),
    //         ]
    //         .into()
    //     );

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v2_1 = g.port_index(v2, 1, Direction::Incoming).unwrap();
    //     let v2_2 = g.port_index(v2, 0, Direction::Outgoing).unwrap();
    //     g.link_ports(v0_0, v2_1).unwrap();
    //     let line = vec![Edge(v0_0, v2_1.into()), Edge(v2_2, None)];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line, TransitionPolicy::NoDanglingEdge),
    //         (4, None)
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v2, PatternNodeAddress(0, 1)),
    //         ]
    //         .into()
    //     );
    // }
}
