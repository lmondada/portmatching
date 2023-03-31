use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Display},
    mem,
};

use portgraph::{dot::dot_string_weighted, NodeIndex, PortGraph, PortIndex, PortOffset, Weights};

use crate::utils::cover::cover_nodes;

use super::address::NodeAddress;

/// A node in the GraphTrie
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `find_port` serves to store the map NodeTransition => PortIndex.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, Default, Debug)]
pub(crate) struct NodeWeight {
    out_port: Option<PortOffset>,
    address: Option<NodeAddress>,
    non_deterministic: bool,
}

impl Display for NodeWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(addr) = &self.address {
            write!(f, "{}", addr)?;
        } else {
            write!(f, "None")?;
        }
        if let Some(port) = &self.out_port {
            write!(f, "[{port:?}]")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateID(NodeIndex);

impl StateID {
    pub fn root() -> Self {
        StateID(NodeIndex::new(0))
    }
}

impl From<NodeIndex> for StateID {
    fn from(value: NodeIndex) -> Self {
        StateID(value)
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PermPortIndex(usize);

impl PermPortIndex {
    fn next(&self) -> Self {
        let PermPortIndex(ind) = self;
        PermPortIndex(ind + 1)
    }
}

pub(crate) struct GraphTrie {
    pub(crate) graph: PortGraph,
    weights: Weights<NodeWeight, StateTransition>,
    perm_indices: RefCell<BTreeMap<PermPortIndex, PortIndex>>,
}

impl Default for GraphTrie {
    fn default() -> Self {
        let graph = Default::default();
        let weights = Default::default();
        let perm_indices = Default::default();
        let mut ret = Self {
            graph,
            weights,
            perm_indices,
        };
        ret.add_state(true);
        ret
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub(crate) enum TrieTraversal {
    ReadOnly,
    Write,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum StateTransition {
    Node(Vec<NodeAddress>, PortOffset),
    NoLinkedNode,
    FAIL,
}

impl Default for StateTransition {
    fn default() -> Self {
        Self::FAIL
    }
}

impl Display for StateTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateTransition::Node(addrs, port) => {
                let addrs: Vec<_> = addrs.iter().map(|addr| addr.to_string()).collect();
                write!(f, "{}[{:?}]", addrs.join("/"), port)
            }
            StateTransition::NoLinkedNode => write!(f, "dangling"),
            StateTransition::FAIL => write!(f, "fail"),
        }
    }
}

impl StateTransition {
    fn transition_type(&self) -> usize {
        match &self {
            StateTransition::Node(_, _) => 0,
            StateTransition::NoLinkedNode => 1,
            StateTransition::FAIL => 2,
        }
    }
}

impl GraphTrie {
    /// An empty graph trie
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn weight(&self, state: StateID) -> &NodeWeight {
        &self.weights[state.0]
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
    pub(crate) fn get_transitions(
        &self,
        state: StateID,
        graph: &PortGraph,
        root: NodeIndex,
        mode: TrieTraversal,
    ) -> Vec<StateTransition> {
        match mode {
            TrieTraversal::ReadOnly => {
                let mut transitions = self
                    .compatible_transitions(state, graph, root, mode)
                    .map(|out_p| self.weights[out_p].clone());
                if self.weight(state).non_deterministic {
                    transitions.collect()
                } else {
                    transitions.next().into_iter().collect()
                }
            }
            TrieTraversal::Write => {
                let &NodeWeight {
                    out_port: offset,
                    address: ref addr,
                    ..
                } = self.weight(state);
                let out_port = addr.as_ref().and_then(|addr| {
                    let node = addr.get_node(graph, root)?;
                    graph.port_index(node, offset?)
                });
                let next_node_port = out_port.and_then(|out_port| {
                    let in_port = graph.port_link(out_port)?;
                    Some((
                        graph.port_node(in_port).expect("invalid port"),
                        graph.port_offset(in_port).expect("invalid port"),
                    ))
                });
                let next_addr = next_node_port.map(|(n, _)| NodeAddress::new(n, graph, root));
                let next_port = next_node_port.map(|(_, p)| p);
                let mut transitions = Vec::new();
                if !self.weight(state).non_deterministic {
                    transitions.extend(self.compatible_transitions(state, graph, root, mode).map(
                        |out_p| {
                            let mut transition = self.weights[out_p].clone();
                            if let (StateTransition::Node(addrs, _), Some(new_addr)) =
                                (&mut transition, &next_addr)
                            {
                                if !addrs.contains(&new_addr) {
                                    addrs.push(new_addr.clone());
                                }
                            }
                            transition
                        },
                    ));
                }
                // Append the perfect transition
                if let (Some(addr), Some(port)) = (next_addr, next_port) {
                    let ideal = StateTransition::Node(vec![addr], port);
                    if !transitions.contains(&ideal) {
                        transitions.push(ideal);
                    }
                } else if out_port.is_some() {
                    let ideal = StateTransition::NoLinkedNode;
                    if !transitions.contains(&ideal) {
                        transitions.push(ideal);
                    }
                }
                transitions
            }
        }
    }

    pub(crate) fn next_states(
        &self,
        state: StateID,
        graph: &PortGraph,
        root: NodeIndex,
    ) -> Vec<StateID> {
        // Compute "ideal" transition
        self.get_transitions(state, graph, root, TrieTraversal::ReadOnly)
            .into_iter()
            .map(|transition| {
                self.transition(state, &transition)
                    .expect("invalid transition")
            })
            .collect()
    }

    /// Split states so that all incoming links are within ports
    ///
    /// Ports are all incoming ports -- split states so that a state
    /// is incident to one of the port iff all ports are within `ports`
    pub(crate) fn create_owned_states<F>(&mut self, mut clone_state: F) -> Vec<StateID>
    where
        F: FnMut(StateID, StateID),
    {
        let all_ports: BTreeSet<_> = self
            .all_perm_ports()
            .into_iter()
            .map(|p| self.free(p).expect("Invalid port"))
            .collect();
        let weights = RefCell::new(&mut self.weights);
        cover_nodes(
            &mut self.graph,
            all_ports,
            |state, new_state, graph| {
                let mut weights = weights.borrow_mut();
                weights[new_state] = weights[state].clone();
                // update transition pointers
                for (out_port, new_out_port) in graph.outputs(state).zip(graph.outputs(new_state)) {
                    weights[new_out_port] = weights[out_port].clone();
                }
                // callback
                clone_state(StateID(state), StateID(new_state));
            },
            |old, new| {
                let mut weights = weights.borrow_mut();
                if let Some(new) = new {
                    weights[new] = mem::take(&mut weights[old]);
                } else {
                    weights[old] = Default::default();
                }
                for val in self.perm_indices.borrow_mut().values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
            },
        )
        .into_iter()
        .map(StateID)
        .collect()
    }

    fn add_state(&mut self, non_deterministic: bool) -> StateID {
        let node = self.graph.add_node(0, 0);
        self.weights[node].non_deterministic = non_deterministic;
        node.into()
    }

    fn add_edge(
        &mut self,
        out_port: PortIndex,
        in_node: StateID,
    ) -> Result<PortIndex, portgraph::LinkError> {
        self.set_num_ports(
            in_node,
            self.graph.num_inputs(in_node.0) + 1,
            self.graph.num_outputs(in_node.0),
        );
        let in_port = self.graph.inputs(in_node.0).last().expect("just created");
        self.graph.link_ports(out_port, in_port).map(|_| in_port)
    }

    fn compatible_transitions<'a>(
        &'a self,
        state: StateID,
        graph: &'a PortGraph,
        root: NodeIndex,
        mode: TrieTraversal,
    ) -> impl Iterator<Item = PortIndex> + 'a {
        let addr = self.weight(state).address.as_ref();
        let offset = self.weight(state).out_port;
        let node = addr.and_then(|addr| addr.get_node(graph, root));
        let out_port = node.and_then(|node| graph.port_index(node, offset?));
        let in_port = out_port.and_then(|out_port| graph.port_link(out_port));
        let in_offset = in_port.and_then(|in_port| graph.port_offset(in_port));
        let next_node = in_port.and_then(|in_port| graph.port_node(in_port));
        self.graph
            .outputs(state.0)
            .filter(move |&out_p| match self.weights[out_p] {
                StateTransition::Node(ref addrs, offset) => {
                    in_offset == Some(offset)
                        && addrs.iter().all(|addr| {
                            let Some(addr_node) = addr.get_node(graph, root) else {
                                // In write mode, being unable to access the node is allowed,
                                // in read-only it is not
                                return mode == TrieTraversal::Write
                            };
                            if let Some(next_node) = next_node {
                                addr_node == next_node
                            } else {
                                false
                            }
                        })
                }
                StateTransition::NoLinkedNode => {
                    // In read mode, out_port existing is enough
                    // In write mode, out_port must be dangling
                    out_port.is_some() && (mode == TrieTraversal::ReadOnly || in_port.is_none())
                }
                StateTransition::FAIL => {
                    // In read mode, FAIL is always an option
                    // In write mode only if the port does not exist
                    mode == TrieTraversal::ReadOnly || out_port.is_none()
                }
            })
    }

    fn transition(&self, state: StateID, transition: &StateTransition) -> Option<StateID> {
        self.graph
            .outputs(state.0)
            .find(|&p| &self.weights[p] == transition)
            .and_then(|p| {
                let in_p = self.graph.port_link(p)?;
                let n = self.graph.port_node(in_p).expect("invalid port");
                Some(StateID(n))
            })
    }

    fn get_or_insert(
        &mut self,
        state: StateID,
        transition: &StateTransition,
        new_state: &mut Option<StateID>,
    ) -> PermPortIndex {
        if !matches!(
            transition,
            StateTransition::FAIL | StateTransition::NoLinkedNode
        ) {
            panic!("can only insert FAIL or NoLinkedNode");
        }
        let out_port = self
            .graph
            .outputs(state.0)
            .find(|&p| &self.weights[p] == transition)
            .unwrap_or_else(|| {
                self.set_num_ports(
                    state,
                    self.graph.num_inputs(state.0),
                    self.graph.num_outputs(state.0) + 1,
                );
                let out_port = self.graph.outputs(state.0).last().expect("just created");
                self.weights[out_port] = transition.clone();
                out_port
            });
        let in_port = self.graph.port_link(out_port).unwrap_or_else(|| {
            let &mut new_state = new_state.get_or_insert_with(|| self.add_state(true));
            self.add_edge(out_port, new_state)
                .expect("out_port is linked?")
        });
        self.create_perm_port(in_port)
    }

    fn get_or_create_start_states(
        &mut self,
        out_port: PortIndex,
        state: StateID,
        graph: &PortGraph,
        root: NodeIndex,
        new_start_state: &mut Option<StateID>,
    ) -> Vec<StateID> {
        let mut start_states = Vec::new();
        let start_node = graph.port_node(out_port).expect("invalid port");
        let start_offset = graph.port_offset(out_port).expect("invalid port");
        let mut curr_states: VecDeque<_> = [state].into();
        while let Some(state) = curr_states.pop_front() {
            let &NodeWeight {
                out_port: offset,
                address: ref addr,
                ..
            } = self.weight(state);
            let offset = offset.unwrap_or(start_offset);
            let addr = addr
                .clone()
                .unwrap_or_else(|| NodeAddress::new(start_node, graph, root));
            if offset == start_offset && addr.get_node(graph, root) == Some(start_node) {
                self.weights[state.0].out_port = Some(offset);
                self.weights[state.0].address = Some(addr);
                start_states.push(state);
            } else {
                if !self.weight(state).non_deterministic {
                    curr_states.extend(
                        self.graph
                            .outputs(state.0)
                            // Filter out FAIL as we add it below anyway
                            .filter(|&p| self.weights[p] != StateTransition::FAIL)
                            .filter_map(|p| Some(self.create_perm_port(self.graph.port_link(p)?)))
                            .map(|p| self.port_state(p).expect("invalid port")),
                    );
                }
                let in_port = self.get_or_insert(state, &StateTransition::FAIL, new_start_state);
                curr_states.push_back(self.port_state(in_port).expect("invalid port"));
            }
        }
        start_states
    }

    /// Add transition while preserving the trie invariants
    ///
    /// We need to add edge leaving from out_port
    ///
    /// `new_state` and `new_start_state` are two optional states that are
    /// free to be used when adding the transition -- these can be used to
    /// avoids creating unnecessary states.
    ///
    /// Steps:
    /// 1. Find state(s) in the future of `state` that ask about out_port
    /// 2. Find which transition to insert
    /// 3. Space out existing transitions to leave space for new ones
    /// 4. Add transitions, return perm ports
    pub(crate) fn add_transition(
        &mut self,
        out_port: PortIndex,
        state: StateID,
        graph: &PortGraph,
        root: NodeIndex,
        new_state: &mut Option<StateID>,
        new_start_state: &mut Option<StateID>,
    ) -> Vec<PermPortIndex> {
        let start_states =
            self.get_or_create_start_states(out_port, state, graph, root, new_start_state);
        let mut in_ports = Vec::new();
        for state in start_states {
            let transitions = self.get_transitions(state, graph, root, TrieTraversal::Write);
            let mut transitions_iter = transitions.iter().peekable();
            match transitions_iter.peek() {
                Some(StateTransition::FAIL) => panic!("invalid start state"),
                Some(StateTransition::NoLinkedNode) => {
                    debug_assert_eq!(transitions.len(), 1);
                    if !self.weight(state).non_deterministic {
                        in_ports.extend(
                            self.graph
                                .outputs(state.0)
                                .filter(|&out_p| {
                                    matches!(
                                        &self.weights[out_p],
                                        StateTransition::FAIL | StateTransition::NoLinkedNode
                                    )
                                })
                                .filter_map(|out_p| self.graph.port_link(out_p))
                                .map(|in_p| self.create_perm_port(in_p)),
                        );
                    }
                    in_ports.push(self.get_or_insert(
                        state,
                        &StateTransition::NoLinkedNode,
                        new_state,
                    ));
                }
                Some(StateTransition::Node(_, _)) => {
                    // Scan the transitions to figure out where and how much we need to shift
                    // it's important that get_transitions returns the transitions in the
                    // port ordering

                    // A vector of indices where new transitions should be inserted, along with
                    // the transition to be inserted
                    let mut new_transitions = Vec::new();
                    for port in self.graph.outputs(state.0) {
                        let curr_transition = &self.weights[port];
                        let port_offset =
                            self.graph.port_offset(port).expect("invalid port").index();
                        if transitions_iter.peek().is_none() {
                            break;
                        }
                        // Insert transitions until the type of transitions match
                        while let Some(transition) = transitions_iter.peek() {
                            if transition.transition_type() < curr_transition.transition_type() {
                                // insert trans_id here
                                new_transitions
                                    .push((port_offset, transitions_iter.next().unwrap().clone()));
                            } else {
                                break;
                            }
                        }
                        // Use existing transition if it's identical
                        if Some(curr_transition) == transitions_iter.peek().copied() {
                            transitions_iter.next();
                        }
                        // Finally, insert new transitions before the current transition
                        // for Node(_, _)s, if curr_transition is strictly contained
                        while let Some(transition) = transitions_iter.peek() {
                            if 'is_compatible: {
                                let (
                    StateTransition::Node(curr_addrs, curr_port),
                    StateTransition::Node(addrs, port),
                ) = (curr_transition, transition) else { break 'is_compatible false };
                                curr_port == port
                                    && curr_addrs.into_iter().all(|addr| addrs.contains(&addr))
                            } {
                                new_transitions
                                    .push((port_offset, transitions_iter.next().unwrap().clone()));
                            } else {
                                break;
                            }
                        }
                    }
                    // Insert remaining transitions at the end
                    let n_ports = self.graph.num_outputs(state.0);
                    new_transitions.extend(transitions_iter.map(|t| (n_ports, t.clone())));

                    // Now shift ports to make space for new ones
                    self.set_num_ports(
                        state,
                        self.graph.num_inputs(state.0),
                        self.graph.num_outputs(state.0) + new_transitions.len(),
                    );

                    let mut new_offset = self.graph.num_outputs(state.0);
                    for offset in (0..=n_ports).rev() {
                        // move existing transition
                        if offset < n_ports && new_offset - 1 > offset {
                            new_offset -= 1;
                            let old = self
                                .graph
                                .port_index(state.0, PortOffset::new_outgoing(offset))
                                .expect("invalid offset");
                            let new = self
                                .graph
                                .port_index(state.0, PortOffset::new_outgoing(new_offset))
                                .expect("invalid offset");
                            self.move_out_port(old, new);
                        }
                        // Insert new transitions
                        while matches!(new_transitions.last(), Some(&(o, _)) if offset == o) {
                            let (_, transition) = new_transitions.pop().expect("just checked");
                            new_offset -= 1;
                            let new = self
                                .graph
                                .port_index(state.0, PortOffset::new_outgoing(new_offset))
                                .expect("invalid offset");
                            self.weights[new] = transition;
                        }
                    }

                    // Finally, return all matching transition ports
                    in_ports.extend(
                        self.graph
                            .outputs(state.0)
                            .filter(|&out_port| transitions.contains(&self.weights[out_port]))
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|out_port| {
                                let in_port = self.graph.port_link(out_port).unwrap_or_else(|| {
                                    let &mut new_node =
                                        new_state.get_or_insert_with(|| self.add_state(false));
                                    self.add_edge(out_port, new_node).expect("out_port free")
                                });
                                self.create_perm_port(in_port)
                            }),
                    );
                }
                None => {}
            }
        }
        in_ports
    }

    fn set_num_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        self.graph
            .set_num_ports(state.0, incoming, outgoing, |old, new| {
                rekey(
                    &mut self.weights,
                    &mut self.perm_indices.borrow_mut(),
                    old,
                    new,
                )
            });
    }

    fn move_out_port(&mut self, old: PortIndex, new: PortIndex) {
        if let Some(in_port) = self.graph.unlink_port(old) {
            self.graph.link_ports(new, in_port).unwrap();
        }
        rekey(
            &mut self.weights,
            &mut self.perm_indices.borrow_mut(),
            old,
            Some(new),
        )
    }

    fn create_perm_port(&self, port: PortIndex) -> PermPortIndex {
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

    // pub(crate) fn reset_perm_ports(&mut self) {
    //     self.perm_indices = Default::default();
    // }

    fn all_perm_ports(&self) -> Vec<PermPortIndex> {
        self.perm_indices.borrow().keys().copied().collect()
    }

    fn port_state(&self, port: PermPortIndex) -> Option<StateID> {
        let &port = self.perm_indices.borrow().get(&port)?;
        self.graph.port_node(port).map(|p| StateID(p))
    }

    pub(crate) fn free(&self, port: PermPortIndex) -> Option<PortIndex> {
        self.perm_indices.borrow_mut().remove(&port)
    }

    pub(crate) fn dotstring(&self) -> String {
        dot_string_weighted(&self.graph, &self.weights)
    }
}

fn rekey<K: Clone, V: Clone + Default>(
    weights: &mut Weights<K, V>,
    perm_indices: &mut BTreeMap<PermPortIndex, PortIndex>,
    old: PortIndex,
    new: Option<PortIndex>,
) {
    if let Some(new) = new {
        weights[new] = mem::take(&mut weights[old]);
    } else {
        weights[old] = Default::default();
    }
    for val in perm_indices.values_mut() {
        if &old == val {
            *val = new.expect("A linked port was deleted!");
        }
    }
}

// impl ReadGraphTrie for GraphTrie {
//     type StateID = StateID;

//     type MatchObject = ();

//     fn init(&self, root: NodeIndex) -> (Self::StateID, Self::MatchObject) {
//         (StateID::ROOT, ())
//     }

//     fn next_states(
//         &self,
//         &state: &Self::StateID,
//         graph: &PortGraph,
//         current_match: &Self::MatchObject,
//     ) -> Vec<(Self::StateID, Self::MatchObject)> {
//         // Compute "ideal" transition
//         let mut next_states: Vec<_> = self
//             .get_transitions(state, graph, current_match, TrieTraversal::ReadOnly)
//             .into_iter()
//             .filter_map(|(transition, current_match)| {
//                 Some((self.transition(state, &transition)?, current_match))
//             })
//             .collect();

//         // Add epsilon transition if we are at a root (i.e. non-deterministic)
//         if self.is_root(state) {
//             if let Some(next_state) = self.transition(state, &NodeTransition::Fail) {
//                 next_states.push((next_state, current_match.clone()));
//             }
//         }
//         next_states
//     }
// }

// impl WriteGraphTrie for GraphTrie {
//     fn create_next_states<F: FnMut(Self::StateID, Self::StateID)>(
//         &mut self,
//         states: Vec<(Self::StateID, Self::MatchObject)>,
//         &Edge(out_port, _): &Edge,
//         graph: &PortGraph,
//         mut clone_state: F,
//     ) -> Vec<(Self::StateID, Self::MatchObject)> {
//         // Technically unnecessary
//         self.reset_perm_ports();

//         // Find the states that correspond to out_port
//         let prev_states: Vec<_> = states.iter().map(|(state, _)| *state).collect();
//         let states = self.valid_start_states(states, out_port, graph);

//         self.create_owned_states(prev_states, self.all_perm_ports(), &mut clone_state);
//         let prev_states: Vec<_> = states.iter().map(|(state, _)| *state).collect();

//         let mut next_states = BTreeSet::new();
//         let mut new_node = None;
//         for (state, current_match) in states {
//             // For every allowable transition, insert it if it does not exist
//             // and return it
//             let (next_ports, next_matches): (Vec<_>, Vec<_>) = self
//                 .get_transition(state, graph, &current_match, TrieTraversal::Write)
//                 .into_iter()
//                 .flat_map(|(mut transition, next_match)| {
//                     if let NodeTransition::NoLinkedNode = &transition {
//                         self.carriage_return(state, current_match.clone(), &mut new_node, true)
//                     } else {
//                         self.expand_addrs(&mut transition, state, graph, &current_match);
//                         [(
//                             self.transition_port(state, &transition).unwrap_or_else(|| {
//                                 let next_addr = next_match.current_addr.clone();
//                                 let next_state = self.extend_tree(
//                                     state,
//                                     next_addr,
//                                     &mut new_node,
//                                     matches!(transition, NodeTransition::NewNode(_)),
//                                 );
//                                 let port = self
//                                     .set_transition(state, next_state, transition)
//                                     .expect("Changing existing value");
//                                 self.create_perm_port(port)
//                             }),
//                             next_match,
//                         )]
//                         .into()
//                     }
//                 })
//                 .unzip();
//             next_states.extend(
//                 next_ports
//                     .iter()
//                     .map(|&p| self.get_state(p).expect("Invalid port"))
//                     .zip(next_matches),
//             );
//         }

//         // Finally, wherever not all inputs come from known nodes, split the
//         // state into two
//         self.create_owned_states(prev_states, self.all_perm_ports(), clone_state);

//         // Merge next states
//         merge_states(next_states)
//     }
// }

#[cfg(test)]
mod tests {
    use portgraph::{NodeIndex, PortGraph, PortOffset};

    use crate::matcher::many_patterns::address::NodeAddress;

    use super::GraphTrie;

    fn link(graph: &mut PortGraph, (out_n, out_p): (usize, usize), (in_n, in_p): (usize, usize)) {
        let out_n = NodeIndex::new(out_n);
        let in_n = NodeIndex::new(in_n);
        let out_p = graph
            .port_index(out_n, PortOffset::new_outgoing(out_p))
            .unwrap();
        let in_p = graph
            .port_index(in_n, PortOffset::new_incoming(in_p))
            .unwrap();
        graph.link_ports(out_p, in_p).unwrap();
    }

    #[test]
    fn test_add_transition_deterministic() {
        let mut trie = GraphTrie::new();
        let state = trie.add_state(false);

        // graph 1
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 0), (1, 0));
        link(&mut graph, (0, 1), (1, 1));

        trie.weights[state.0].address = NodeAddress::new(root, &graph, root).into();
        trie.weights[state.0].out_port = PortOffset::new_outgoing(1).into();

        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));

        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state.0), 3);
        assert_eq!(trie.graph.node_count(), 4);

        // graph 3
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();
        link(&mut graph, (0, 0), (1, 1));

        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state.0), 4);
        assert_eq!(trie.graph.node_count(), 6);

        // graph 4
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();

        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        assert_eq!(trie.graph.node_count(), 7);
    }

    #[test]
    fn test_add_transition_non_deterministic() {
        let mut trie = GraphTrie::new();
        let state = trie.add_state(true);

        // graph 1
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 0), (1, 0));
        link(&mut graph, (0, 1), (1, 1));
        trie.weights[state.0].address = NodeAddress::new(root, &graph, root).into();
        trie.weights[state.0].out_port = PortOffset::new_outgoing(1).into();

        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));
        trie.add_transition(out_port, state, &graph, root, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state.0), 2);
        assert_eq!(trie.graph.node_count(), 4);
    }
}
