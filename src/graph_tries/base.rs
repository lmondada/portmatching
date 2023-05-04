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
    utils::cover::untangle_threads,
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
        if self.non_deterministic {
            write!(f, "<font color=\"red\">")?;
        }
        if let Some(addr) = &self.address {
            write!(f, "{:?}", addr)?;
        } else {
            write!(f, "None")?;
        }
        if let Some(port) = &self.out_port {
            write!(f, "[{port:?}]")?;
        }
        if self.non_deterministic {
            write!(f, "</font>")?;
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

    // The following are only useful during construction
    pub(super) perm_indices: RefCell<BTreeMap<PermPortIndex, (PortIndex, usize)>>,
    pub(super) edge_cnt: usize,
    pub(super) new_in_ports: BTreeSet<PermPortIndex>,
    pub(super) start_states: BTreeMap<StateID, Vec<usize>>,
}

impl Default for BaseGraphTrie<(Vec<PortOffset>, usize)> {
    fn default() -> Self {
        let graph = Default::default();
        let weights = Default::default();
        let perm_indices = Default::default();
        let new_in_ports = Default::default();
        let start_states = Default::default();
        let mut ret = Self {
            graph,
            weights,
            perm_indices,
            edge_cnt: 0,
            new_in_ports,
            start_states,
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

    /// Try to convert into a non-deterministic state.
    ///
    /// If the state is already non-deterministic, this is a no-op. If it is
    /// deterministic and has more than one transition, then the conversion
    /// fails.
    ///
    /// Returns whether the state is non-deterministic after the conversion.
    #[allow(clippy::wrong_self_convention)]
    fn into_non_deterministic(&mut self, state: NodeIndex) -> bool {
        let det_flag = &mut self.weights[state].non_deterministic;
        if self.graph.num_outputs(state) <= 1 {
            *det_flag = true;
        }
        *det_flag
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

    pub(crate) fn gen_transitions(
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

    /// Reorganise trie after having added transitions.
    ///
    /// This step is essential after each pattern that has been added to the trie.
    /// It splits states where necessary so that there is no "cross-talk", i.e.
    /// none of the transitions added will form a shortcut in the trie.
    ///
    /// In the process, it might clone states. To keep track of the identity
    /// of the states, the caller should pass a callback function that will be
    /// called for each state that is cloned.
    pub fn finalize<F>(&mut self, mut clone_state: F)
    where
        F: FnMut(StateID, StateID),
    {
        // Reset all trackers
        self.edge_cnt = 0;
        let start_states = mem::take(&mut self.start_states);
        let new_in_ports: BTreeSet<_> = mem::take(&mut self.new_in_ports)
            .into_iter()
            .map(|p| self.to_port(p).expect("invalid port"))
            .collect();

        // All the threads that have been created during this pattern
        let all_threads: BTreeSet<_> = self
            .all_perm_ports()
            .into_iter()
            .map(|p| self.free(p).expect("Invalid port"))
            .collect();
        let weights = RefCell::new(&mut self.weights);


        untangle_threads(
            &mut self.graph,
            all_threads,
            &new_in_ports,
            &start_states,
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
                for (val, _) in self.perm_indices.borrow_mut().values_mut() {
                    if &old == val {
                        *val = new.expect("A linked port was deleted!");
                    }
                }
            },
        );
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
    ) -> Result<PermPortIndex, portgraph::LinkError> {
        self.set_num_ports(
            in_node,
            self.graph.num_inputs(in_node) + 1,
            self.graph.num_outputs(in_node),
        );
        let in_port = self.graph.inputs(in_node).last().expect("just created");
        self.graph.link_ports(out_port, in_port)?;
        let ret = self.create_perm_port(in_port);
        self.new_in_ports.insert(ret);
        Ok(ret)
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

    /// Follow FAIL transition, creating a new state if necessary.
    fn follow_fail(&mut self, state: StateID, new_state: &mut Option<StateID>) -> PermPortIndex {
        let last_port = self.graph.outputs(state).last();
        if let Some(last_port) = last_port {
            if self.weights[last_port] == StateTransition::FAIL {
                let in_port = self
                    .graph
                    .port_link(last_port)
                    .expect("Disconnected transition");
                return self.create_perm_port(in_port);
            }
        }
        self.append_transition(state, new_state, StateTransition::FAIL)
    }

    /// Follow NoLinkedNode transition, creating a new state if necessary.
    ///
    /// This requires shifting FAIL transition if it exists
    fn follow_no_linked_node(
        &mut self,
        state: StateID,
        new_state: &mut Option<StateID>,
    ) -> PermPortIndex {
        let num_outputs = self.graph.num_outputs(state);
        let last_port = num_outputs
            .checked_sub(1)
            .map(|o| self.graph.output(state, o).expect("valid offset"));
        let prev_port = num_outputs
            .checked_sub(2)
            .map(|o| self.graph.output(state, o).expect("valid offset"));
        let last_weight = last_port.map(|p| &self.weights[p]);
        let prev_weight = prev_port.map(|p| &self.weights[p]);
        match (last_weight, prev_weight) {
            (Some(StateTransition::NoLinkedNode), _) => {
                // Follow existing NoLinkedNode transition
                let out_port = last_port.expect("last_weight is Some");
                let in_port = self
                    .graph
                    .port_link(out_port)
                    .expect("Disconnected transition");
                self.create_perm_port(in_port)
            }
            (_, Some(StateTransition::NoLinkedNode)) => {
                // Follow existing NoLinkedNode transition
                let out_port = prev_port.expect("prev_weight is Some");
                let in_port = self
                    .graph
                    .port_link(out_port)
                    .expect("Disconnected transition");
                self.create_perm_port(in_port)
            }
            (Some(StateTransition::Node(_, _)), _) | (None, _) => {
                // Add a NoLinkedNode transition at the end
                self.append_transition(state, new_state, StateTransition::NoLinkedNode)
            }
            (Some(StateTransition::FAIL), _) => {
                // The trickiest case: add port, then move FAIL to the end
                self.set_num_ports(
                    state,
                    self.graph.num_inputs(state),
                    self.graph.num_outputs(state) + 1,
                );
                // Move FAIL to make space for NoLinkedNode
                let fail_offset = self.graph.num_outputs(state) - 1;
                let dangling_offset = fail_offset - 1;
                let fail = self
                    .graph
                    .output(state, fail_offset)
                    .expect("offset < num_outputs");
                let dangling = self
                    .graph
                    .output(state, dangling_offset)
                    .expect("offset < num_outputs");
                self.move_out_port(dangling, fail);
                // Add NoLinkedNode weight
                self.weights[dangling] = StateTransition::NoLinkedNode;
                // Link falls back to FAIL transition
                let next_p = self.graph.port_link(fail).expect("Disconnected transition");
                let next = self.graph.port_node(next_p).expect("invalid port");
                self.add_edge(dangling, next)
                    .expect("freed by move_out_port")
            }
        }
    }

    /// Append transition at the end of `state`
    ///
    /// Careful! The order of transitions is very important and appending at the
    /// end without checking the ordering is incorrect
    fn append_transition(
        &mut self,
        state: StateID,
        new_state: &mut Option<StateID>,
        transition: EdgeWeight,
    ) -> PermPortIndex {
        self.set_num_ports(
            state,
            self.graph.num_inputs(state),
            self.graph.num_outputs(state) + 1,
        );
        let last_port = self.graph.outputs(state).last().expect("just created");
        self.weights[last_port] = transition;
        let next = new_state.get_or_insert_with(|| self.add_state(false));
        self.add_edge(last_port, *next).expect("just created")
    }

    fn valid_start_states(
        &mut self,
        graph_edge: PortIndex,
        trie_state: StateID,
        skeleton: &Skeleton,
        deterministic: bool,
        new_start_state: &mut Option<StateID>,
    ) -> Vec<StateID> {
        let mut start_states = Vec::new();
        let mut curr_states: VecDeque<_> = [trie_state].into();
        while let Some(state) = curr_states.pop_front() {
            // Try to convert to start state
            if self.into_start_state(state, graph_edge, skeleton, deterministic) {
                start_states.push(state);
            } else {
                // Not a start state, so follow all possible edges and start over
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
                let in_port = self.follow_fail(state, new_start_state);
                curr_states.push_back(self.port_state(in_port).expect("invalid port"));
            }
        }
        start_states
    }

    /// Add graph edge to the trie.
    ///
    /// The new trie states created can be either deterministic or not, as
    /// controlled by the `non_deterministic` parameter. Non-deterministic
    /// states have worse matching performance, but too many deterministic nodes
    /// can lead to a large trie.
    ///
    /// Important: you must call [`Self::finalize`] every time after having added
    /// all the edges of a pattern. Otherwise the trie will not be valid.
    ///
    /// Returns the trie states after the edge has been added.
    pub fn add_graph_edge(
        &mut self,
        edge: PortIndex,
        trie_states: impl IntoIterator<Item = StateID>,
        deterministic: bool,
        skeleton: &Skeleton,
    ) -> BTreeSet<StateID> {
        // 1. Find trie states that can be used as start states, i.e. states
        // whose address matches the source node of the edge
        let mut new_start_state = None;
        let start_states = trie_states
            .into_iter()
            .flat_map(|state| {
                self.valid_start_states(edge, state, skeleton, deterministic, &mut new_start_state)
                    .into_iter()
            })
            .collect::<BTreeSet<_>>();

        // Record states as start states
        for &state in &start_states {
            self.start_states.entry(state).or_default().push(self.edge_cnt);
        }
        // Increase edge count when we've found the new start state
        self.edge_cnt += 1;

        // 2. For each start state, add the edge to the trie
        let mut new_state = None;
        let mut in_ports = Vec::new();
        for state in start_states {
            let transitions = self.gen_transitions(state, skeleton);
            in_ports.append(&mut self.insert_transitions(state, transitions, &mut new_state));
        }

        in_ports
            .into_iter()
            .map(|p| self.port_state(p).expect("invalid port"))
            .collect()
    }

    /// Add graph edge to the trie using deterministic strategy.
    pub fn add_graph_edge_det(
        &mut self,
        edge: PortIndex,
        trie_states: impl IntoIterator<Item = StateID>,
        skeleton: &Skeleton,
    ) -> BTreeSet<StateID> {
        self.add_graph_edge(edge, trie_states, true, skeleton)
    }

    /// Add graph edge to the trie using non-deterministic strategy.
    pub fn add_graph_edge_nondet(
        &mut self,
        edge: PortIndex,
        trie_states: impl IntoIterator<Item = StateID>,
        skeleton: &Skeleton,
    ) -> BTreeSet<StateID> {
        self.add_graph_edge(edge, trie_states, false, skeleton)
    }

    /// Insert transitions respecting the StateTransition ordering.
    ///
    /// Transitions at every state are ordered by the PartialOrd on
    /// StateTransition. A transition is strictly greater when it is more
    /// general: a transition condition that is satisfied will also satisfy
    /// any strictly greater transition.
    ///
    /// Note that within [`StateTransition::Node`], not all transitions are
    /// comparable and so the ordering is not unique. It is given by the order
    /// in which transitions were inserted and will always be preserved by future
    /// calls to [`Self::insert_transitions`].
    ///
    /// It is important that `transitions` is ordered in the same order as
    /// the ports.
    fn insert_transitions(
        &mut self,
        state: NodeIndex,
        transitions: impl IntoIterator<Item = StateTransition<(Address<usize>, Vec<Rib>)>>,
        new_state: &mut Option<NodeIndex>,
    ) -> Vec<PermPortIndex> {
        let mut in_ports = Vec::new();
        // The transitions, along with the index where they should be inserted
        let mut new_transitions = Vec::new();
        let mut offset = 0;
        'transition: for transition in transitions {
            let transition = transition.into_simplified();
            match transition {
                StateTransition::FAIL => panic!("invalid start state"),
                StateTransition::NoLinkedNode => {
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
                    in_ports.push(self.follow_no_linked_node(state, new_state));
                    // We can return early because there can only be one transition
                    return in_ports;
                }
                StateTransition::Node(_, _) => {}
            }
            // Advance to the first transition that is strictly more general
            // If such a transition never comes then add it at the end
            let (curr_port, curr_transition) = loop {
                let Some(curr_port) = self.graph.output(state, offset) else {
                    // Add transition at the end
                    new_transitions.push((offset, transition));
                    continue 'transition
                };
                let curr_transition = &self.weights[curr_port];
                if !curr_transition.ge(&transition) {
                    offset += 1;
                } else {
                    break (curr_port, curr_transition);
                }
            };
            if curr_transition == &transition {
                // Use existing transition if it's identical
                let in_port = self
                    .graph
                    .port_link(curr_port)
                    .expect("Disconnected transition");
                in_ports.push(self.create_perm_port(in_port));
            } else {
                // Otherwise insert new transition here
                new_transitions.push((offset, transition));
            }
        }

        // Create new ports for the new transitions
        let n_ports = self.graph.num_outputs(state);
        self.set_num_ports(
            state,
            self.graph.num_inputs(state),
            n_ports + new_transitions.len(),
        );

        // Now shift ports to make space for new ones

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
                let next_state = if !self.weights[state].non_deterministic {
                    fallback
                } else {
                    None
                }
                .unwrap_or_else(|| *new_state.get_or_insert_with(|| self.add_state(false)));
                let in_port = self.add_edge(new, next_state).expect("new port index");
                in_ports.push(in_port);
            }
            if offset == new_offset {
                // There are no more empty slots. We are done
                break;
            }
        }
        in_ports
    }

    /// Try to convert into a start state for `graph_edge`
    #[allow(clippy::wrong_self_convention)]
    fn into_start_state(
        &mut self,
        trie_state: StateID,
        graph_edge: PortIndex,
        skeleton: &Skeleton,
        deterministic: bool,
    ) -> bool {
        let graph = skeleton.graph();
        let start_node = graph.port_node(graph_edge).expect("invalid port");
        let start_offset = graph.port_offset(graph_edge).expect("invalid port");

        let trie_offset = self.weight(trie_state).out_port.unwrap_or(start_offset);
        let mut fallback_spine = None;
        let spine = self.spine(trie_state).unwrap_or_else(|| {
            fallback_spine = Some(skeleton.get_spine());
            fallback_spine.as_ref().unwrap()
        });
        let graph_addr = {
            let addressing = PortGraphAddressing::new(skeleton.root(), graph, Some(spine), None);
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
        let spine = self.spine(trie_state).unwrap_or_else(|| {
            fallback_spine = Some(skeleton.get_spine());
            fallback_spine.as_ref().unwrap()
        });
        let trie_addr = self
            .weight(trie_state)
            .address
            .unwrap_or_else(|| graph_addr.expect("Could not get address of current node"));
        if trie_offset == start_offset && Some(&trie_addr) == graph_addr.as_ref() {
            self.weights[trie_state].spine = Some(spine.to_vec());
            self.weights[trie_state].out_port = Some(trie_offset);
            self.weights[trie_state].address = Some(trie_addr);
            if !deterministic {
                // Try to convert state into a non-deterministic one
                self.into_non_deterministic(trie_state);
            }
            true
        } else {
            false
        }
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
        self.perm_indices
            .borrow_mut()
            .insert(next_ind, (port, self.edge_cnt));
        next_ind
    }

    fn all_perm_ports(&self) -> Vec<PermPortIndex> {
        self.perm_indices.borrow().keys().copied().collect()
    }

    fn port_state(&self, port: PermPortIndex) -> Option<StateID> {
        let port = self.to_port(port)?;
        self.graph.port_node(port)
    }

    fn to_port(&self, port: PermPortIndex) -> Option<PortIndex> {
        let &(port, _) = self.perm_indices.borrow().get(&port)?;
        Some(port)
    }

    pub(crate) fn free(&self, port: PermPortIndex) -> Option<(PortIndex, usize)> {
        self.perm_indices.borrow_mut().remove(&port)
    }

    fn spine(&self, state: StateID) -> Option<&[(Vec<PortOffset>, usize)]> {
        self.weight(state).spine.as_deref()
    }

    pub(crate) fn to_cached_trie(&self) -> BaseGraphTrie<(SpineID, Vec<PortOffset>, usize)> {
        BaseGraphTrie::<(SpineID, Vec<PortOffset>, usize)>::new(self)
    }
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
    perm_indices: &mut BTreeMap<PermPortIndex, (PortIndex, usize)>,
    old: PortIndex,
    new: Option<PortIndex>,
) {
    if let Some(new) = new {
        weights[new] = mem::take(&mut weights[old]);
    } else {
        weights[old] = Default::default();
    }
    for (val, _) in perm_indices.values_mut() {
        if &old == val {
            *val = new.expect("A linked port was deleted!");
        }
    }
}

#[cfg(test)]
mod tests {
    use portgraph::{NodeIndex, PortGraph, PortOffset};

    use crate::addressing::Skeleton;

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
        trie.weights[state].address = Some((0, 0));
        trie.weights[state].out_port = PortOffset::new_outgoing(1).into();

        trie.add_graph_edge_det(out_port, [state], &skel);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));
        let skel = Skeleton::new(&graph, root);

        trie.add_graph_edge_det(out_port, [state], &skel);

        assert_eq!(trie.graph.num_outputs(state), 3);
        assert_eq!(trie.graph.node_count(), 4);

        // graph 3
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();
        link(&mut graph, (0, 0), (1, 1));

        let skel = Skeleton::new(&graph, root);
        trie.add_graph_edge_det(out_port, [state], &skel);

        assert_eq!(trie.graph.num_outputs(state), 4);
        assert_eq!(trie.graph.node_count(), 6);

        // graph 4
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 1);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(0)).unwrap();
        let skel = Skeleton::new(&graph, root);

        trie.add_graph_edge_det(out_port, [state], &skel);

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

        trie.add_graph_edge_nondet(out_port, [state], &skel);

        // graph 2
        let mut graph = PortGraph::new();
        let root = graph.add_node(0, 3);
        graph.add_node(2, 0);
        let out_port = graph.port_index(root, PortOffset::new_outgoing(1)).unwrap();
        link(&mut graph, (0, 1), (1, 1));
        let skel = Skeleton::new(&graph, root);
        trie.add_graph_edge_nondet(out_port, [state], &skel);

        assert_eq!(trie.graph.num_outputs(state), 2);
        assert_eq!(trie.graph.node_count(), 4);
    }
}
