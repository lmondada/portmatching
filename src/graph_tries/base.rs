use std::{
    cell::RefCell,
    collections::{BTreeSet, VecDeque},
    fmt::{self, Debug, Display},
    mem,
};

use portgraph::{
    dot::dot_string_weighted, Direction, NodeIndex, PortGraph, PortIndex,
    PortOffset, UnmanagedDenseMap, Weights,
};

use crate::{constraint::Constraint, utils::cover::untangle_threads};

use super::{optimise::get_next_world_age, GraphTrie, StateID};

/// A node in the GraphTrie.
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `find_port` serves to store the map NodeTransition => PortIndex.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct NodeWeight<A> {
    pub(crate) out_port: Option<A>,
    pub(crate) non_deterministic: bool,
}

impl<A> Default for NodeWeight<A> {
    fn default() -> Self {
        Self {
            out_port: Default::default(),
            non_deterministic: Default::default(),
        }
    }
}

impl<A: Debug> Display for NodeWeight<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.non_deterministic {
            write!(f, "<font color=\"red\">")?;
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

// impl<'n, T: SpineAddress> NodeWeight<T> {
//     fn address(&'n self) -> Option<Address<T::AsRef<'n>>> {
//         let (spine_ind, ind) = self.address?;
//         let spine = self.spine.as_ref()?[spine_ind].as_ref();
//         Some((spine, ind))
//     }
// }

type EdgeWeight<C> = Option<C>;

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
///
/// As transitions are added we also keep track of how the trie is traversed,
/// forming the "trace" of the trie. This is used in [`Self::finalize`] to
/// split states where necessary to keep avoid cross-talk, i.e. creating new
/// disallowed state transitions as a by-product of the new transitions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaseGraphTrie<C, A> {
    pub(crate) graph: PortGraph,
    pub(crate) weights: Weights<NodeWeight<A>, EdgeWeight<C>>,

    // The following are only useful during construction
    pub(super) trace: UnmanagedDenseMap<PortIndex, (Vec<usize>, bool)>,
    pub(super) world_age: usize,
}

impl<C, A> Debug for BaseGraphTrie<C, A>
where
    A: Debug,
    C: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaseGraphTrie")
            .field("graph", &self.graph)
            .field("weights", &self.weights)
            .finish()
    }
}

impl<C: Clone, A: Clone> Clone for BaseGraphTrie<C, A> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            weights: self.weights.clone(),
            trace: self.trace.clone(),
            world_age: self.world_age,
        }
    }
}

impl<C: Clone + Ord + Constraint, A: Clone + Ord> Default for BaseGraphTrie<C, A> {
    fn default() -> Self {
        let graph = Default::default();
        let weights = Default::default();
        let trace = Default::default();
        let mut ret = Self {
            graph,
            weights,
            trace,
            world_age: 0,
        };
        ret.add_state(true);
        ret
    }
}

impl<C: Clone, A: Clone> GraphTrie for BaseGraphTrie<C, A> {
    type Constraint = C;
    type Address = A;

    fn trie(&self) -> &PortGraph {
        &self.graph
    }

    fn port_address(&self, state: StateID) -> Option<&A> {
        self.weights[state].out_port.as_ref()
    }

    fn transition(&self, port: PortIndex) -> Option<&Self::Constraint> {
        self.weights[port].as_ref()
    }

    fn is_non_deterministic(&self, state: StateID) -> bool {
        self.weights[state].non_deterministic
    }
}

impl<C: Display + Clone, A: Debug + Clone> BaseGraphTrie<C, A> {
    pub(crate) fn str_weights(&self) -> Weights<String, String> {
        let mut str_weights = Weights::new();
        for p in self.graph.ports_iter() {
            str_weights[p] = match self.graph.port_direction(p).unwrap() {
                Direction::Incoming => "".to_string(),
                Direction::Outgoing => self.weights[p]
                    .as_ref()
                    .map(|c| c.to_string())
                    .unwrap_or("FAIL".to_string()),
            };
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

impl<C: Clone + Ord + Constraint, A: Clone + Ord> BaseGraphTrie<C, A> {
    /// An empty graph trie
    pub fn new() -> Self {
        Self::default()
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

    pub(crate) fn weight(&self, state: StateID) -> &NodeWeight<A> {
        &self.weights[state]
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
    pub fn finalize<F>(&mut self, root: NodeIndex, mut clone_state: F) -> BTreeSet<NodeIndex>
    where
        F: FnMut(StateID, StateID),
    {
        // Reset all trackers
        self.world_age = 0;
        let trace = mem::take(&mut self.trace);

        let weights = RefCell::new(&mut self.weights);

        untangle_threads(
            &mut self.graph,
            trace,
            root,
            |state, new_state, graph| {
                let mut weights = weights.borrow_mut();
                weights[new_state] = weights[state].clone();
                // update transition pointers
                for out_port in graph.outputs(state) {
                    let offset = graph.port_offset(out_port).expect("invalid port");
                    let new_out_port = graph.port_index(new_state, offset).expect("invalid port");
                    weights[new_out_port] = weights[out_port].clone();
                }
                // callback
                clone_state(state, new_state);
            },
            |old, new| {
                let mut weights = weights.borrow_mut();
                weights.ports.rekey(old, new);
                self.trace.rekey(old, new)
            },
        )
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
        let out_node = self.graph.port_node(out_port).expect("invalid port");
        if out_node == in_node {
            panic!("adding cyclic edge");
        }
        let unlinked_port = self
            .graph
            .inputs(in_node)
            .find(|&p| self.graph.port_link(p).is_none());
        let in_port = unlinked_port.unwrap_or_else(|| {
            self.set_num_ports(
                in_node,
                self.graph.num_inputs(in_node) + 1,
                self.graph.num_outputs(in_node),
            );
            self.graph.inputs(in_node).last().expect("just created")
        });
        self.graph.link_ports(out_port, in_port)?;
        self.trace[out_port].1 = true;
        self.trace[in_port].1 = true;
        Ok(in_port)
    }

    /// Follow FAIL transition, creating a new state if necessary.
    pub(super) fn follow_fail(
        &mut self,
        state: StateID,
        new_state: &mut Option<StateID>,
        from_world_age: usize,
        to_world_age: usize,
    ) -> PortIndex {
        let fail_port = self.graph.outputs(state).find(|&p| {
            self.weights[p].is_none()
                && (!self.trace[p].1 || self.trace[p].0.contains(&from_world_age))
        });
        let (out_port, in_port) = if let Some(out_port) = fail_port {
            let in_port = self
                .graph
                .port_link(out_port)
                .expect("Disconnected transition");
            (out_port, in_port)
        } else {
            self.append_transition(state, new_state, None)
        };
        if !self.trace[out_port].0.contains(&from_world_age) {
            self.trace[out_port].0.push(from_world_age);
            self.trace[in_port].0.push(to_world_age);
        }
        in_port
    }

    /// Append transition at the end of `state`
    ///
    /// Careful! The order of transitions is very important and appending at the
    /// end without checking the ordering is incorrect
    fn append_transition(
        &mut self,
        state: StateID,
        new_state: &mut Option<StateID>,
        constraint: EdgeWeight<C>,
    ) -> (PortIndex, PortIndex) {
        self.set_num_ports(
            state,
            self.graph.num_inputs(state),
            self.graph.num_outputs(state) + 1,
        );
        let last_port = self.graph.outputs(state).last().expect("just created");
        self.weights[last_port] = constraint;
        let next = new_state.get_or_insert_with(|| self.add_state(false));
        let in_port = self.add_edge(last_port, *next).expect("just created");
        (last_port, in_port)
    }

    pub(super) fn valid_start_states(
        &mut self,
        out_port: &A,
        trie_state: StateID,
        deterministic: bool,
        new_start_state: &mut Option<StateID>,
        world_age: usize,
    ) -> Vec<StateID> {
        let mut start_states = Vec::new();
        let mut curr_states: VecDeque<_> = [trie_state].into();
        while let Some(state) = curr_states.pop_front() {
            // Try to convert to start state
            if self.into_start_state(state, out_port, deterministic) {
                start_states.push(state);
            } else {
                // Not a start state, so follow all possible edges and start over
                if !self.weight(state).non_deterministic {
                    for out_port in self.graph.outputs(state) {
                        if self.weights[out_port].is_none() {
                            // Filter out FAIL as we add it below anyway
                            continue;
                        }
                        let in_port = self.graph.port_link(out_port).expect("Disconnected edge");
                        let node = self.graph.port_node(in_port).expect("invalid port");
                        if !self.trace[out_port].0.contains(&world_age) {
                            self.trace[out_port].0.push(world_age);
                            self.trace[in_port].0.push(world_age);
                            curr_states.push_back(node);
                        }
                    }
                }
                let in_port = self.follow_fail(state, new_start_state, world_age, world_age);
                curr_states.push_back(self.graph.port_node(in_port).expect("invalid port"));
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
        out_port: &A,
        trie_states: impl IntoIterator<Item = StateID>,
        deterministic: bool,
        constraint: C,
    ) -> BTreeSet<StateID> {
        // 1. Find trie states that can be used as start states, i.e. states
        // whose address matches the source node of the edge
        let mut new_start_state = None;
        let start_states = trie_states
            .into_iter()
            .flat_map(|state| {
                self.valid_start_states(
                    out_port,
                    state,
                    deterministic,
                    &mut new_start_state,
                    self.world_age,
                )
                .into_iter()
            })
            .collect::<BTreeSet<_>>();

        // 2. For each start state, add the edge to the trie
        let mut new_state = None;
        let mut next_states = BTreeSet::new();
        for state in start_states {
            next_states.extend(
                self.insert_transitions(
                    state,
                    constraint.clone(),
                    &mut new_state,
                    self.world_age,
                    self.world_age + 1,
                )
                .into_iter(),
            );
        }

        // Increase edge count for next state
        self.world_age += 1;

        next_states
    }

    /// Add graph edge to the trie using deterministic strategy.
    pub fn add_graph_edge_det(
        &mut self,
        edge: &A,
        trie_states: impl IntoIterator<Item = StateID>,
        constraint: C,
    ) -> BTreeSet<StateID> {
        self.add_graph_edge(edge, trie_states, true, constraint)
    }

    /// Add graph edge to the trie using non-deterministic strategy.
    pub fn add_graph_edge_nondet(
        &mut self,
        edge: &A,
        trie_states: impl IntoIterator<Item = StateID>,
        constraint: C,
    ) -> BTreeSet<StateID> {
        self.add_graph_edge(edge, trie_states, false, constraint)
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
    pub(super) fn insert_transitions(
        &mut self,
        state: NodeIndex,
        new_cond: C,
        new_state: &mut Option<NodeIndex>,
        from_world_age: usize,
        to_world_age: usize,
    ) -> Vec<NodeIndex> {
        self.insert_transitions_filtered(
            state,
            new_cond,
            new_state,
            |_| true,
            from_world_age,
            to_world_age,
        )
        .0
    }

    pub(super) fn insert_transitions_ages(
        &mut self,
        state: NodeIndex,
        new_cond: C,
        new_state: &mut Option<NodeIndex>,
        from_world_age: usize,
        to_world_age: usize,
    ) -> (Vec<NodeIndex>, Vec<usize>) {
        let (a, _, b) = self.insert_transitions_filtered(
            state,
            new_cond,
            new_state,
            |_| true,
            from_world_age,
            to_world_age,
        );
        (a, b)
    }

    pub(super) fn insert_transitions_filtered<F>(
        &mut self,
        state: NodeIndex,
        new_cond: C,
        new_state: &mut Option<NodeIndex>,
        mut transition_filter: F,
        from_world_age: usize,
        to_world_age: usize,
    ) -> (Vec<NodeIndex>, Vec<C>, Vec<usize>)
    where
        F: FnMut(&C) -> bool,
    {
        // The states we are transitioning to, to be returned
        let mut next_states = Vec::new();
        let mut used_transitions = Vec::new();
        let mut next_world_ages = Vec::new();

        // The transitions, along with the index where they should be inserted
        let mut new_transitions = Vec::new();
        let mut alread_inserted = BTreeSet::new();
        let mut offset = 0;

        // Compute the transitions to add
        loop {
            let Some(transition) = self.graph.output(state, offset) else {
                // We passed the last transition: insert and stop iteration
                if alread_inserted.insert(new_cond.clone()) {
                    new_transitions.push((offset, new_cond));
                }
                break;
            };
            let Some(curr_cond) = self.weights[transition].as_ref() else {
                // FAIL transition: insert before and stop iteration
                if alread_inserted.insert(new_cond.clone()) {
                    new_transitions.push((offset, new_cond));
                }
                break;
            };
            if !transition_filter(curr_cond) {
                // We ignore transitions that do not pass the filter
                offset += 1;
                continue;
            }
            let Some(merged_cond) = curr_cond.and(&new_cond) else {
                // We ignore conditions we cannot merge with
                offset += 1;
                continue
            };
            if &merged_cond != curr_cond {
                if !self.weights[state].non_deterministic {
                    // insert new condition before current one
                    if alread_inserted.insert(merged_cond.clone()) {
                        new_transitions.push((offset, merged_cond.clone()));
                    }
                } else {
                    // Just insert the new condition and that's it
                    if alread_inserted.insert(new_cond.clone()) {
                        new_transitions.push((offset, new_cond));
                    }
                    break;
                }
            } else if (!self.weights[state].non_deterministic || curr_cond == &new_cond)
                && (!self.trace[transition].1 || self.trace[transition].0.contains(&from_world_age))
            {
                // use existing transition
                let in_port = self
                    .graph
                    .port_link(transition)
                    .expect("Disconnected transition");
                if !self.trace[transition].0.contains(&from_world_age) {
                    self.trace[transition].0.push(from_world_age);
                    self.trace[in_port].0.push(to_world_age);
                }
                next_states.push(self.graph.port_node(in_port).expect("invalid port"));
                used_transitions.push(new_cond.clone());
                next_world_ages.push(get_next_world_age(
                    transition,
                    in_port,
                    &self.trace,
                    from_world_age,
                ));
                alread_inserted.insert(curr_cond.clone());
            } else if !self.weights[state].non_deterministic || curr_cond == &new_cond {
                // Copy existing transition
                if alread_inserted.insert(merged_cond.clone()) {
                    new_transitions.push((offset, merged_cond.clone()));
                }
            }
            if merged_cond == new_cond {
                // we've inserted the new condition, our job is done
                break;
            }
            offset += 1;
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
                self.weights[new] = Some(transition.clone());
                let next_state = if !self.weights[state].non_deterministic {
                    fallback
                } else {
                    None
                }
                .unwrap_or_else(|| *new_state.get_or_insert_with(|| self.add_state(false)));
                let in_port = self.add_edge(new, next_state).expect("new port index");
                self.trace[new].0.push(from_world_age);
                self.trace[in_port].0.push(to_world_age);
                next_states.push(next_state);
                used_transitions.push(transition);
                next_world_ages.push(to_world_age)
            }
            if offset == new_offset {
                // There are no more empty slots. We are done
                break;
            }
        }
        (next_states, used_transitions, next_world_ages)
    }

    /// Try to convert into a start state for `graph_edge`
    #[allow(clippy::wrong_self_convention)]
    fn into_start_state(&mut self, trie_state: StateID, out_port: &A, deterministic: bool) -> bool {
        // let start_node = graph.port_node(graph_edge).expect("invalid port");
        // let start_offset = graph.port_offset(graph_edge).expect("invalid port");

        let trie_out_port = self
            .weight(trie_state)
            .out_port
            .as_ref()
            .unwrap_or(out_port);
        if trie_out_port == out_port {
            self.weights[trie_state].out_port = Some(trie_out_port.clone());
            if !deterministic {
                // Try to convert state into a non-deterministic one
                self.into_non_deterministic(trie_state);
            }
            true
        } else {
            false
        }
    }

    pub(super) fn set_num_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        if state == NodeIndex::new(2) {
            println!(
                "N4: ({}, {}) => ({incoming}, {outgoing})",
                self.graph.num_inputs(state),
                self.graph.num_outputs(state)
            );
        }
        self.graph
            .set_num_ports(state, incoming, outgoing, |old, new| {
                println!("{old:?} => {new:?}");
                let new = new.new_index();
                self.trace.rekey(old, new);
                self.weights.ports.rekey(old, new);
            });
    }

    fn move_out_port(&mut self, old: PortIndex, new: PortIndex) {
        if let Some(in_port) = self.graph.unlink_port(old) {
            self.graph.link_ports(new, in_port).unwrap();
        }
        self.trace.rekey(old, Some(new));
        self.weights.ports.rekey(old, Some(new));
    }
}
