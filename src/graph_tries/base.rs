use std::{
    cell::RefCell,
    cmp,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Display},
    mem, vec,
};

use portgraph::{
    dot::dot_string_weighted, Direction, NodeIndex, PortGraph, PortIndex, PortOffset, Weights,
};

use crate::{
    addressing::{
        cache::SpineID, pg::AsPathOffset, Address, AsSpineID, PortGraphAddressing, Rib, Skeleton,
        SkeletonAddressing, Spine, SpineAddress,
    },
    utils::cover::cover_nodes,
};

use super::{GraphTrie, StateID, StateTransition};

/// A node in the GraphTrie.
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `find_port` serves to store the map NodeTransition => PortIndex.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, Default, Debug)]
pub(crate) struct NodeWeight<T> {
    pub(crate) out_port: Option<PortOffset>,
    pub(crate) address: Option<Address<usize>>,
    pub(crate) spine: Option<Spine<T>>,
    pub(crate) non_deterministic: bool,
}

impl<T> Display for NodeWeight<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(addr) = &self.address {
            write!(f, "{:?}", addr)?;
        } else {
            write!(f, "None")?;
        }
        if let Some(port) = &self.out_port {
            write!(f, "[{port:?}]")?;
        }
        Ok(())
    }
}

impl<'n, T: SpineAddress> NodeWeight<T> {
    fn address(&'n self) -> Option<Address<T::AsRef<'n>>> {
        let (spine_ind, ind) = self.address?;
        let spine = self.spine.as_ref()?[spine_ind].as_ref();
        Some((spine, ind))
    }
}

type EdgeWeight = StateTransition<(Address<usize>, Vec<Rib>)>;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PermPortIndex(usize);

impl PermPortIndex {
    fn next(&self) -> Self {
        let PermPortIndex(ind) = self;
        PermPortIndex(ind + 1)
    }
}

/// A graph trie implementation using portgraph.
///
/// The parameter T is the type of the spine. Any type that implements
/// [`SpineAddress`] can be used as spine in theory, however constructing such
/// a trie is only supported for very specific types.
///
/// The construction of such tries is the most complex logic in this
/// crate.
///
/// There are deterministic and non-deterministic states, describing how
/// the state automaton in that state should behave.
/// Roughly speaking, the role of deterministic and non-deterministic states is
/// inverted when writing to the trie (as opposed to reading):
///  * a deterministic state at read time means that every possible transition must
///    be considered at write time (as we do not know which one will be taken)
///  * a non-deterministic state at read time on the other hand is guaranteed to
///    follow any allowed transition, and thus at write time we can just follow
///    one of the transitions.
///
/// The deterministic states are thus the harder states to handle at write time.
/// The main idea is that transitions are considered from left to right, so when
/// adding a new transition, all the transitions to its right might get
/// "shadowed" by it. That means that we must make sure that if an input graph
/// chooses the new transition over the old one, then all the patterns that would
/// have been discovered along the old path will still be discovered.
///
/// Similarly, transitions to the left of a new transition can shadow the new
/// transition, so we must make sure that if an input graph chooses one of these
/// higher priority transition, it still discovers the pattern that is being added.
#[derive(Clone, Debug)]
pub struct BaseGraphTrie<T = (Vec<PortOffset>, usize)> {
    pub(crate) graph: PortGraph,
    pub(crate) weights: Weights<NodeWeight<T>, EdgeWeight>,
    pub(super) perm_indices: RefCell<BTreeMap<PermPortIndex, PortIndex>>,
}

impl Default for BaseGraphTrie<(Vec<PortOffset>, usize)> {
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

impl SpineAddress for (Vec<PortOffset>, usize) {
    type AsRef<'n> = (&'n [PortOffset], usize);

    fn as_ref(&self) -> Self::AsRef<'_> {
        (self.0.as_slice(), self.1)
    }
}

impl<S> GraphTrie for BaseGraphTrie<S>
where
    S: SpineAddress + Clone,
    for<'n> S::AsRef<'n>: Copy + Default + PartialEq + AsPathOffset + AsSpineID,
{
    type Addressing<'g, 'n> = PortGraphAddressing<'g, 'n, S> where S: 'n;
    type SpineID = S;

    fn trie(&self) -> &PortGraph {
        &self.graph
    }

    fn address(&self, state: StateID) -> Option<Address<S::AsRef<'_>>> {
        self.weights[state].address()
    }

    fn port_offset(&self, state: StateID) -> Option<PortOffset> {
        self.weights[state].out_port
    }

    fn transition<'g, 'n, 'm: 'n>(
        &'n self,
        port: PortIndex,
        addressing: &Self::Addressing<'g, 'm>,
    ) -> StateTransition<(Self::Addressing<'g, 'n>, Address<S::AsRef<'n>>)> {
        let node = self.graph.port_node(port).expect("invalid port");
        let spine = self.spine(node);

        match &self.weights[port] {
            StateTransition::Node(addrs, port) => StateTransition::Node(
                addrs
                    .iter()
                    .map(|(addr, ribs)| {
                        let &(spine_ind, ind) = addr;
                        let vert = &spine.expect("address with no spine")[spine_ind];
                        let addr = (vert.as_ref(), ind);
                        (addressing.with_spine(spine.unwrap()).with_ribs(ribs), addr)
                    })
                    .collect(),
                *port,
            ),
            StateTransition::NoLinkedNode => StateTransition::NoLinkedNode,
            StateTransition::FAIL => StateTransition::FAIL,
        }
    }

    fn is_non_deterministic(&self, state: StateID) -> bool {
        self.weights[state].non_deterministic
    }

    fn spine(&self, state: StateID) -> Option<&Vec<S>> {
        self.weights[state].spine.as_ref()
    }
}

impl<T: Clone> BaseGraphTrie<T> {
    pub(crate) fn str_weights(&self) -> Weights<String, String> {
        let mut str_weights = Weights::new();
        for p in self.graph.ports_iter() {
            str_weights[p] = match self.graph.port_direction(p).unwrap() {
                Direction::Incoming => "".to_string(),
                Direction::Outgoing => self.weights[p].to_string(),
            }
        }
        for n in self.graph.nodes_iter() {
            str_weights[n] = self.weights[n].to_string();
        }
        str_weights
    }

    pub(crate) fn _dotstring(&self) -> String {
        dot_string_weighted(&self.graph, &self.str_weights())
    }
}

impl BaseGraphTrie<(Vec<PortOffset>, usize)> {
    /// An empty graph trie
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn weight(&self, state: StateID) -> &NodeWeight<(Vec<PortOffset>, usize)> {
        &self.weights[state]
    }

    pub(crate) fn get_transitions_write(
        &mut self,
        state: StateID,
        skeleton: &Skeleton,
    ) -> Vec<StateTransition<(Address<usize>, Vec<Rib>)>> {
        let addressing =
            PortGraphAddressing::new(skeleton.root(), skeleton.graph(), self.spine(state), None);
        let next_node = self.next_node(state, &addressing, &mut ());
        let next_port = self.next_port_offset(state, &addressing, &mut ());
        let next_addr = next_node.map(|n| {
            let addressing = PortGraphAddressing::new(
                skeleton.root(),
                skeleton.graph(),
                self.spine(state),
                None,
            );
            addressing
                .get_addr(n, &mut ())
                .map(|((path, offset), ind)| {
                    let line_ind = self.weights[state]
                        .spine
                        .as_ref()
                        .expect("next_node exists")
                        .iter()
                        .position(|(p, o)| p.as_slice() == path && *o == offset)
                        .expect("could not find address in spine");
                    (line_ind, ind)
                })
                .unwrap_or_else(|| {
                    let NodeWeight {
                        out_port,
                        address,
                        spine,
                        ..
                    } = &mut self.weights[state];
                    let spine = spine.as_mut().expect("next_node exists");
                    let (line_ind, _) = skeleton.extend_spine(
                        spine,
                        address.as_ref().expect("next_node exists"),
                        out_port.expect("next_node exists"),
                    );
                    let ind = match out_port.unwrap().direction() {
                        Direction::Incoming => -1,
                        Direction::Outgoing => 1,
                    };
                    (line_ind, ind)
                })
        });
        let spine = self.spine(state);
        let next_ribs = spine.and_then(|spine| {
            let mut ribs = skeleton.get_ribs(spine);
            update_ribs(&mut ribs, *next_addr.as_ref()?);
            Some(ribs)
        });
        let mut transitions = Vec::new();
        let addressing =
            PortGraphAddressing::new(skeleton.root(), skeleton.graph(), self.spine(state), None);
        if !self.weight(state).non_deterministic {
            transitions.extend(self.compatible_transitions(state, &addressing).filter_map(
                |out_p| {
                    let mut transition = self.weights[out_p].clone();
                    if let (StateTransition::Node(addrs, _), Some(new_addr)) =
                        (&mut transition, &next_addr)
                    {
                        let merged_addrs = merge_addrs(
                            addrs.as_slice(),
                            &[(*new_addr, next_ribs.clone().unwrap())],
                        )?;
                        if !is_satisfied(next_node?, &merged_addrs, &addressing, spine?) {
                            return None;
                        }
                        *addrs = merged_addrs;
                    }
                    Some(transition)
                },
            ));
        }
        // Append the perfect transition
        if let (Some(addr), Some(port)) = (next_addr, next_port) {
            let ideal = StateTransition::Node(
                vec![(addr, next_ribs.clone().expect("next_addr != None"))],
                port,
            );
            if !transitions.contains(&ideal) {
                transitions.push(ideal);
            }
        } else if self.port(state, &addressing, &mut ()).is_some() {
            let ideal = StateTransition::NoLinkedNode;
            if !transitions.contains(&ideal) {
                transitions.push(ideal);
            }
        }
        transitions
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
                clone_state(state, new_state);
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
        .collect()
    }

    fn add_state(&mut self, non_deterministic: bool) -> StateID {
        let node = self.graph.add_node(0, 0);
        self.weights[node].non_deterministic = non_deterministic;
        node
    }

    fn add_edge(
        &mut self,
        out_port: PortIndex,
        in_node: StateID,
    ) -> Result<PortIndex, portgraph::LinkError> {
        self.set_num_ports(
            in_node,
            self.graph.num_inputs(in_node) + 1,
            self.graph.num_outputs(in_node),
        );
        let in_port = self.graph.inputs(in_node).last().expect("just created");
        self.graph.link_ports(out_port, in_port).map(|_| in_port)
    }

    /// All transitions in `state` that are allowed for `graph`
    fn compatible_transitions<'g: 'n, 'n>(
        &'n self,
        state: StateID,
        addressing: &'g PortGraphAddressing<'g, 'n, (Vec<PortOffset>, usize)>,
    ) -> impl Iterator<Item = PortIndex> + 'n {
        let graph = addressing.graph();
        let out_port = self.port(state, addressing, &mut ());
        let in_port = out_port.and_then(|out_port| graph.port_link(out_port));
        let in_offset = in_port.map(|in_port| graph.port_offset(in_port).expect("invalid port"));
        let next_node = in_port.map(|in_port| graph.port_node(in_port).expect("invalid port"));
        self.graph
            .outputs(state)
            .filter(move |&out_p| match self.weights[out_p] {
                StateTransition::Node(ref addrs, offset) => {
                    if in_offset != Some(offset) {
                        return false;
                    }
                    addrs.iter().all(|(addr, _)| {
                        let &(line_ind, ind) = addr;
                        let (ref path, offset) =
                            self.spine(state).expect("invalid state")[line_ind];
                        let addr = ((path.as_slice(), offset), ind);
                        let Some(node) = addressing.get_node(
                                &addr,
                                &mut ()
                            ) else {
                                return true
                            };
                        node == next_node.expect("from if condition")
                    })
                }
                StateTransition::NoLinkedNode => {
                    // In write mode, out_port must be dangling
                    out_port.is_some() && in_port.is_none()
                }
                StateTransition::FAIL => {
                    // In write mode only if the port does not exist
                    out_port.is_none()
                }
            })
    }

    fn get_or_insert(
        &mut self,
        state: StateID,
        transition: &StateTransition<(Address<usize>, Vec<Rib>)>,
        new_state: &mut Option<StateID>,
    ) -> PermPortIndex {
        if matches!(transition, StateTransition::Node(_, _)) {
            panic!("can only insert FAIL or NoLinkedNode")
        };
        let out_port = self
            .graph
            .outputs(state)
            .find(|&p| &self.weights[p] == transition)
            .unwrap_or_else(|| {
                self.set_num_ports(
                    state,
                    self.graph.num_inputs(state),
                    self.graph.num_outputs(state) + 1,
                );
                let mut offset = self.graph.outputs(state).len() - 1;
                // shift transitions until out_port is in the right place
                loop {
                    if offset == 0 {
                        break;
                    }
                    let prev_port = self
                        .graph
                        .output(state, offset - 1)
                        .expect("0 <= offset < len");
                    if self.weights[prev_port].transition_type() > transition.transition_type() {
                        let new = self.graph.output(state, offset).expect("0 <= offset < len");
                        self.move_out_port(prev_port, new);
                    } else {
                        break;
                    }
                    offset -= 1;
                }
                let out_port = self.graph.output(state, offset).expect("0 <= offset < len");
                self.weights[out_port] = transition.clone();
                out_port
            });
        let in_port = self.graph.port_link(out_port).unwrap_or_else(|| {
            let offset = self
                .graph
                .port_offset(out_port)
                .expect("invalid port")
                .index();
            let next_state = self
                .graph
                .output(state, offset + 1)
                .and_then(|p| self.graph.port_link(p))
                .map(|p| self.graph.port_node(p).expect("invalid port"))
                .unwrap_or_else(|| *new_state.get_or_insert_with(|| self.add_state(true)));
            self.add_edge(out_port, next_state)
                .expect("out_port is linked?")
        });
        self.create_perm_port(in_port)
    }

    fn get_or_create_start_states(
        &mut self,
        out_port: PortIndex,
        state: StateID,
        skeleton: &Skeleton,
        new_start_state: &mut Option<StateID>,
    ) -> Vec<StateID> {
        let graph = skeleton.graph();
        let mut start_states = Vec::new();
        let start_node = graph.port_node(out_port).expect("invalid port");
        let start_offset = graph.port_offset(out_port).expect("invalid port");
        let mut curr_states: VecDeque<_> = [state].into();
        while let Some(state) = curr_states.pop_front() {
            let offset = self.weight(state).out_port.unwrap_or(start_offset);
            let mut fallback_spine = None;
            let spine = self.spine(state).unwrap_or_else(|| {
                fallback_spine = Some(skeleton.get_spine());
                fallback_spine.as_ref().unwrap()
            });
            let graph_addr = {
                let addressing =
                    PortGraphAddressing::new(skeleton.root(), skeleton.graph(), Some(spine), None);
                addressing
                    .get_addr(start_node, &mut ())
                    .map(|((path, offset), ind)| {
                        let line_ind = spine
                            .iter()
                            .position(|&(ref a, b)| a == path && b == offset)
                            .expect("Could not find path in spine");
                        (line_ind, ind)
                    })
            };
            let spine = self.spine(state).unwrap_or_else(|| {
                fallback_spine = Some(skeleton.get_spine());
                fallback_spine.as_ref().unwrap()
            });
            let addr = self
                .weight(state)
                .address
                .unwrap_or_else(|| graph_addr.expect("Could not get address of current node"));
            if offset == start_offset && Some(&addr) == graph_addr.as_ref() {
                self.weights[state].spine = Some(spine.to_vec());
                self.weights[state].out_port = Some(offset);
                self.weights[state].address = Some(addr);
                start_states.push(state);
            } else {
                if !self.weight(state).non_deterministic {
                    curr_states.extend(
                        self.graph
                            .outputs(state)
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
        skeleton: &Skeleton,
        new_state: &mut Option<StateID>,
        new_start_state: &mut Option<StateID>,
    ) -> Vec<PermPortIndex> {
        let start_states =
            self.get_or_create_start_states(out_port, state, skeleton, new_start_state);
        let mut in_ports = Vec::new();
        for state in start_states {
            let transitions = self.get_transitions_write(state, skeleton);
            let mut transitions_iter = transitions.iter().peekable();
            match transitions_iter.peek() {
                Some(StateTransition::FAIL) => panic!("invalid start state"),
                Some(StateTransition::NoLinkedNode) => {
                    debug_assert_eq!(transitions.len(), 1);
                    if !self.weight(state).non_deterministic {
                        in_ports.extend(
                            self.graph
                                .outputs(state)
                                .filter(|&out_p| {
                                    !matches!(
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
                    for port in self.graph.outputs(state) {
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
                            if let Some(in_port) = self.graph.port_link(port) {
                                in_ports.push(self.create_perm_port(in_port));
                            }
                            transitions_iter.next();
                        }
                        // Finally, insert new transition if it is strictly stronger
                        // than the current one
                        while let Some(transition) = transitions_iter.peek() {
                            let (
                                StateTransition::Node(curr_addrs, curr_port),
                                StateTransition::Node(addrs, port),
                            ) = (curr_transition, transition) else {
                                break
                            };
                            if curr_port != port {
                                break;
                            }
                            if curr_addrs.iter().all(|(c_addr, c_ribs)| {
                                addrs
                                    .iter()
                                    .any(|(addr, ribs)| addr == c_addr && ribs_within(c_ribs, ribs))
                            }) {
                                new_transitions
                                    .push((port_offset, transitions_iter.next().unwrap().clone()));
                            } else {
                                break;
                            }
                        }
                    }
                    // Insert remaining transitions at the end
                    let n_ports = self.graph.num_outputs(state);
                    new_transitions.extend(transitions_iter.map(|t| (n_ports, t.clone())));
                    // sort transitions
                    new_transitions.sort_by_key(|&(p, _)| p);

                    // Now shift ports to make space for new ones
                    self.set_num_ports(
                        state,
                        self.graph.num_inputs(state),
                        self.graph.num_outputs(state) + new_transitions.len(),
                    );

                    let mut new_offset = self.graph.num_outputs(state);
                    for offset in (0..=n_ports).rev() {
                        // move existing transition
                        let fallback = if offset < n_ports {
                            new_offset -= 1;
                            let old = self
                                .graph
                                .port_index(state, PortOffset::new_outgoing(offset))
                                .expect("invalid offset");
                            let new = self
                                .graph
                                .port_index(state, PortOffset::new_outgoing(new_offset))
                                .expect("invalid offset");
                            self.move_out_port(old, new);
                            self.graph
                                .port_link(new)
                                .map(|p| self.graph.port_node(p).expect("invalid port"))
                        } else {
                            None
                        };
                        // Insert new transitions
                        while matches!(new_transitions.last(), Some(&(o, _)) if offset == o) {
                            let (_, transition) = new_transitions.pop().expect("just checked");
                            new_offset -= 1;
                            let new = self
                                .graph
                                .port_index(state, PortOffset::new_outgoing(new_offset))
                                .expect("invalid offset");
                            self.weights[new] = transition;
                            let next_state = fallback.unwrap_or_else(|| {
                                *new_state.get_or_insert_with(|| self.add_state(false))
                            });
                            let in_port = self.add_edge(new, next_state).expect("new port index");
                            in_ports.push(self.create_perm_port(in_port));
                        }
                        if offset == new_offset {
                            // There are no more empty slots. We are done
                            break;
                        }
                    }
                }
                None => {}
            }
        }
        in_ports
    }

    fn set_num_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        self.graph
            .set_num_ports(state, incoming, outgoing, |old, new| {
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

    fn all_perm_ports(&self) -> Vec<PermPortIndex> {
        self.perm_indices.borrow().keys().copied().collect()
    }

    fn port_state(&self, port: PermPortIndex) -> Option<StateID> {
        let &port = self.perm_indices.borrow().get(&port)?;
        self.graph.port_node(port)
    }

    pub(crate) fn free(&self, port: PermPortIndex) -> Option<PortIndex> {
        self.perm_indices.borrow_mut().remove(&port)
    }

    fn spine(&self, state: StateID) -> Option<&[(Vec<PortOffset>, usize)]> {
        self.weight(state).spine.as_deref()
    }

    pub(crate) fn to_cached_trie(&self) -> BaseGraphTrie<(SpineID, Vec<PortOffset>, usize)> {
        BaseGraphTrie::<(SpineID, Vec<PortOffset>, usize)>::new(self)
    }
}

fn ribs_within(a: &[[isize; 2]], b: &[[isize; 2]]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(a, b)| a[0] >= b[0] && a[1] <= b[1])
}

fn update_ribs(ribs: &mut [[isize; 2]], (line, ind): (usize, isize)) {
    let [min, max] = &mut ribs[line];
    *min = cmp::min(*min, ind);
    *max = cmp::max(*max, ind);
}

fn is_satisfied(
    node: NodeIndex,
    addrs: &[(Address<usize>, Vec<Rib>)],
    addressing: &PortGraphAddressing<'_, '_, (Vec<PortOffset>, usize)>,
    spine: &[(Vec<PortOffset>, usize)],
) -> bool {
    addrs.iter().all(|&((line_ind, ind), _)| {
        let spine = spine[line_ind].as_ref();
        let addr = (spine, ind);
        if let Some(graph_node) = addressing.get_node(&addr, &mut ()) {
            node == graph_node
        } else {
            true
        }
    })
}

fn merge_addrs(
    addrs1: &[(Address<usize>, Vec<Rib>)],
    addrs2: &[(Address<usize>, Vec<Rib>)],
) -> Option<Vec<(Address<usize>, Vec<Rib>)>> {
    let mut addrs = Vec::from_iter(addrs1.iter().cloned());
    for (a, rib_a) in addrs2 {
        let mut already_inserted = false;
        for (b, rib_b) in addrs.iter_mut() {
            if a == b {
                // If they share the same address, then ribs is union of ribs
                for (ra, rb) in rib_a.iter().zip(rib_b.iter_mut()) {
                    rb[0] = cmp::min(ra[0], rb[0]);
                    rb[1] = cmp::max(ra[1], rb[1]);
                }
                already_inserted = true;
            } else {
                // Otherwise the smaller address must be outside larger's ribs
                let (a, rib_b) = if a < b {
                    (a, rib_b as &_)
                } else {
                    (b as &_, rib_a)
                };
                let b_int = rib_b[a.0];
                if b_int[0] <= a.1 && b_int[1] >= a.1 {
                    // Then address A < B would have been chosen as B's address!
                    return None;
                }
            }
        }
        if !already_inserted {
            addrs.push((*a, rib_a.clone()));
        }
    }
    Some(addrs)
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

#[cfg(test)]
mod tests {
    use portgraph::{NodeIndex, PortGraph, PortOffset};

    use crate::addressing::{PortGraphAddressing, Skeleton, SkeletonAddressing};

    use super::BaseGraphTrie;

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
        let mut trie = BaseGraphTrie::<(Vec<PortOffset>, usize)>::new();
        let state = trie.add_state(false);

        // graph 1
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 0), (1, 0));
        link(&mut graph, (0, 1), (1, 1));

        let skel = Skeleton::new(&graph, root);
        let addressing = PortGraphAddressing::from_skeleton(&skel);
        let addr = addressing
            .get_addr(root, &mut ())
            .map(|((path, offset), ind)| {
                let line_ind = skel
                    .get_spine()
                    .iter()
                    .position(|&(ref a, b)| a == path && b == offset)
                    .expect("Could not find path in spine");
                (line_ind, ind)
            });
        trie.weights[state].address = addr;
        trie.weights[state].out_port = PortOffset::new_outgoing(1).into();

        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));
        let skel = Skeleton::new(&graph, root);

        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state), 3);
        assert_eq!(trie.graph.node_count(), 4);

        // graph 3
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();
        link(&mut graph, (0, 0), (1, 1));

        let skel = Skeleton::new(&graph, root);
        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state), 4);
        assert_eq!(trie.graph.node_count(), 6);

        // graph 4
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();
        let skel = Skeleton::new(&graph, root);

        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        assert_eq!(trie.graph.node_count(), 7);
    }

    #[test]
    fn test_add_transition_non_deterministic() {
        let mut trie = BaseGraphTrie::<(Vec<PortOffset>, usize)>::new();
        let state = trie.add_state(true);

        // graph 1
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 0), (1, 0));
        link(&mut graph, (0, 1), (1, 1));
        let skel = Skeleton::new(&graph, root);
        trie.weights[state].address = (0, 0).into();
        trie.weights[state].out_port = PortOffset::new_outgoing(1).into();

        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));
        let skel = Skeleton::new(&graph, root);
        trie.add_transition(out_port, state, &skel, &mut None, &mut None);

        assert_eq!(trie.graph.num_outputs(state), 2);
        assert_eq!(trie.graph.node_count(), 4);
    }
}
