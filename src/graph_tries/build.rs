use std::{
    cell::RefCell,
    collections::{BTreeSet, VecDeque},
    fmt::Debug,
    fmt::Display,
    mem,
};

use portgraph::{
    dot::dot_string_weighted, NodeIndex, PortIndex, PortOffset, UnmanagedDenseMap,
    Weights,
};

use crate::{
    utils::{
        age,
        cover::{untangle_threads, SplitNodesMap},
    },
    Constraint,
};

use super::{get_next_world_age, BaseGraphTrie, EdgeWeight, StateID};

pub struct GraphTrieBuilder<C, A, Age> {
    pub(super) trie: BaseGraphTrie<C, A>,
    pub(super) trace: UnmanagedDenseMap<PortIndex, (Vec<Age>, bool)>,
    pub(super) world_age: Age,
}

impl<C, A, Age: Default + Clone> GraphTrieBuilder<C, A, Age> {
    pub fn new(trie: BaseGraphTrie<C, A>) -> Self {
        Self {
            trie,
            trace: UnmanagedDenseMap::new(),
            world_age: Default::default(),
        }
    }

    pub fn age(&self) -> &Age {
        &self.world_age
    }
}

impl<C: Display + Clone, A: Debug + Clone, Age: Debug + Clone> GraphTrieBuilder<C, A, Age> {
    fn str_weights(&self) -> Weights<String, String> {
        let mut str_weights = self.trie.str_weights();
        for p in self.trie.graph.ports_iter() {
            str_weights[p] += &format!(" [{:?}]", self.trace[p]);
        }
        str_weights
    }

    pub fn dotstring(&self) -> String {
        dot_string_weighted(&self.trie.graph, &self.str_weights())
    }
}

impl<C, A, Age: Default> GraphTrieBuilder<C, A, Age>
where
    A: Clone + Ord,
    C: Clone + Ord + Constraint,
    Age: Clone,
{
    /// Reorganise trie after having added transitions.
    ///
    /// This step is essential after each pattern that has been added to the trie.
    /// It splits states where necessary so that there is no "cross-talk", i.e.
    /// none of the transitions added will form a shortcut in the trie.
    ///
    /// In the process, it might clone states. To keep track of the identity
    /// of the states, the caller should pass a callback function that will be
    /// called for each state that is cloned.
    pub fn finalize<F>(
        self,
        root: NodeIndex,
        mut clone_state: F,
    ) -> (BaseGraphTrie<C, A>, SplitNodesMap<Age>)
    where
        F: FnMut(StateID, StateID),
        Age: Ord,
    {
        // Reset all trackers
        let GraphTrieBuilder {
            mut trie, trace, ..
        } = self;

        let weights = RefCell::new(&mut trie.weights);

        let new_nodes = untangle_threads(
            &mut trie.graph,
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
            },
        );
        (trie, new_nodes)
    }

    pub(super) fn skip_finalize(self) -> BaseGraphTrie<C, A> {
        self.trie
    }

    /// Follow FAIL transition, creating a new state if necessary.
    pub(super) fn follow_fail<F: FnOnce(StateID, &mut BaseGraphTrie<C, A>) -> StateID>(
        &mut self,
        state: StateID,
        new_state: F,
        from_world_age: &Age,
        to_world_age: &Age,
    ) -> PortIndex
    where
        Age: Eq + age::Age,
    {
        let fail_port = self.trie.graph.outputs(state).find(|&p| {
            self.trie.weights[p].is_none()
                && (!self.trace[p].1 || self.trace[p].0.contains(from_world_age))
        });
        let (out_port, in_port) = if let Some(out_port) = fail_port {
            let in_port = self
                .trie
                .graph
                .port_link(out_port)
                .expect("Disconnected transition");
            (out_port, in_port)
        } else {
            self.append_transition(state, new_state, None)
        };
        trace_insert(
            &mut self.trace,
            out_port,
            in_port,
            from_world_age.clone(),
            to_world_age.clone(),
        );
        in_port
    }

    fn add_edge(
        &mut self,
        out_port: PortIndex,
        in_node: StateID,
    ) -> Result<PortIndex, portgraph::LinkError> {
        let out_node = self.trie.graph.port_node(out_port).expect("invalid port");
        if out_node == in_node {
            panic!("adding cyclic edge");
        }
        let unlinked_port = self
            .trie
            .graph
            .inputs(in_node)
            .find(|&p| self.trie.graph.port_link(p).is_none());
        let in_port = unlinked_port.unwrap_or_else(|| {
            self.set_num_ports(
                in_node,
                self.trie.graph.num_inputs(in_node) + 1,
                self.trie.graph.num_outputs(in_node),
            );
            self.trie
                .graph
                .inputs(in_node)
                .last()
                .expect("just created")
        });
        self.trie.graph.link_ports(out_port, in_port)?;
        self.trace[out_port].1 = true;
        self.trace[in_port].1 = true;
        Ok(in_port)
    }

    /// Append transition at the end of `state`
    ///
    /// Careful! The order of transitions is very important and appending at the
    /// end without checking the ordering is incorrect
    fn append_transition<F: FnOnce(StateID, &mut BaseGraphTrie<C, A>) -> StateID>(
        &mut self,
        state: StateID,
        new_state: F,
        constraint: EdgeWeight<C>,
    ) -> (PortIndex, PortIndex) {
        self.set_num_ports(
            state,
            self.trie.graph.num_inputs(state),
            self.trie.graph.num_outputs(state) + 1,
        );
        let last_port = self.trie.graph.outputs(state).last().expect("just created");
        self.trie.weights[last_port] = constraint;
        let new_state = new_state(state, &mut self.trie);
        let in_port = self.add_edge(last_port, new_state).expect("just created");
        (last_port, in_port)
    }

    pub(super) fn valid_start_states<F: FnMut(StateID, &mut BaseGraphTrie<C, A>) -> StateID>(
        &mut self,
        out_port: &A,
        trie_state: StateID,
        deterministic: bool,
        mut new_state: F,
        from_world_age: &Age,
        to_world_age: &Age,
    ) -> Vec<StateID>
    where
        Age: Eq + age::Age,
    {
        let mut start_states = Vec::new();
        let mut curr_states: VecDeque<_> = [(trie_state, from_world_age.clone())].into();
        // let mut world_age = from_world_age;
        while let Some((state, world_age)) = curr_states.pop_front() {
            // Try to convert to start state
            if self.trie.into_start_state(state, out_port, deterministic) {
                start_states.push(state);
            } else {
                // Not a start state, so follow all possible edges and start over
                if !self.trie.weight(state).non_deterministic {
                    for out_port in self.trie.graph.outputs(state) {
                        if self.trie.weights[out_port].is_none() {
                            // Filter out FAIL as we add it below anyway
                            continue;
                        }
                        let in_port = self
                            .trie
                            .graph
                            .port_link(out_port)
                            .expect("Disconnected edge");
                        let node = self.trie.graph.port_node(in_port).expect("invalid port");
                        let next_world_age = trace_insert(
                            &mut self.trace,
                            out_port,
                            in_port,
                            world_age.clone(),
                            to_world_age.clone(),
                        );
                        curr_states.push_back((node, next_world_age.clone()));
                    }
                }
                let in_port = self.follow_fail(state, &mut new_state, &world_age, to_world_age);
                let out_port = self
                    .trie
                    .graph
                    .port_link(in_port)
                    .expect("Disconnected edge");
                let world_age = get_next_world_age(out_port, in_port, &self.trace, &world_age);
                curr_states.push_back((
                    self.trie.graph.port_node(in_port).expect("invalid port"),
                    world_age.clone(),
                ));
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
    ) -> BTreeSet<StateID>
    where
        Age: age::Age + Eq,
    {
        // 1. Find trie states that can be used as start states, i.e. states
        // whose address matches the source node of the edge
        let mut new_start_state = None;
        let world_age = mem::take(&mut self.world_age);
        let start_states = trie_states
            .into_iter()
            .flat_map(|state| {
                self.valid_start_states(
                    out_port,
                    state,
                    deterministic,
                    |_, trie| *new_start_state.get_or_insert_with(|| trie.add_state(false)),
                    &world_age,
                    &world_age,
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
                    |_, trie| *new_state.get_or_insert_with(|| trie.add_state(false)),
                    &world_age,
                    &world_age.next(),
                )
                .into_iter(),
            );
        }

        // Increase edge count for next state
        self.world_age = world_age.next();

        next_states
    }

    /// Add graph edge to the trie using deterministic strategy.
    pub fn add_graph_edge_det(
        &mut self,
        edge: &A,
        trie_states: impl IntoIterator<Item = StateID>,
        constraint: C,
    ) -> BTreeSet<StateID>
    where
        Age: Eq + age::Age,
    {
        self.add_graph_edge(edge, trie_states, true, constraint)
    }

    /// Add graph edge to the trie using non-deterministic strategy.
    pub fn add_graph_edge_nondet(
        &mut self,
        edge: &A,
        trie_states: impl IntoIterator<Item = StateID>,
        constraint: C,
    ) -> BTreeSet<StateID>
    where
        Age: Eq + age::Age,
    {
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
    pub(super) fn insert_transitions<F>(
        &mut self,
        state: NodeIndex,
        new_cond: C,
        new_state: F,
        from_world_age: &Age,
        to_world_age: &Age,
    ) -> Vec<NodeIndex>
    where
        Age: Eq + age::Age,
        F: for<'a> FnMut(StateID, &'a mut BaseGraphTrie<C, A>) -> StateID,
    {
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

    pub(super) fn insert_transitions_filtered<F, G>(
        &mut self,
        state: NodeIndex,
        new_cond: C,
        mut new_state: G,
        mut transition_filter: F,
        from_world_age: &Age,
        to_world_age: &Age,
    ) -> (Vec<NodeIndex>, Vec<C>)
    where
        F: FnMut(&C) -> bool,
        G: FnMut(StateID, &mut BaseGraphTrie<C, A>) -> StateID,
        Age: Eq + age::Age,
    {
        // The states we are transitioning to, to be returned
        let mut next_states = Vec::new();
        let mut used_transitions = Vec::new();

        // The transitions, along with the index where they should be inserted
        let mut new_transitions = Vec::new();
        let mut alread_inserted = BTreeSet::new();
        let mut offset = 0;

        // Compute the transitions to add
        loop {
            let Some(transition) = self.trie.graph.output(state, offset) else {
                // We passed the last transition: insert and stop iteration
                if alread_inserted.insert(new_cond.clone()) {
                    new_transitions.push((offset, new_cond));
                }
                break;
            };
            let Some(curr_cond) = self.trie.weights[transition].as_ref() else {
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
                if !self.trie.weights[state].non_deterministic {
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
            } else if (!self.trie.weights[state].non_deterministic || curr_cond == &new_cond)
                && (!self.trace[transition].1 || self.trace[transition].0.contains(from_world_age))
            {
                // use existing transition
                let in_port = self
                    .trie
                    .graph
                    .port_link(transition)
                    .expect("Disconnected transition");
                trace_insert(
                    &mut self.trace,
                    transition,
                    in_port,
                    from_world_age.clone(),
                    to_world_age.clone(),
                );
                next_states.push(self.trie.graph.port_node(in_port).expect("invalid port"));
                used_transitions.push(new_cond.clone());
                alread_inserted.insert(curr_cond.clone());
            } else if !self.trie.weights[state].non_deterministic || curr_cond == &new_cond {
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
        let n_ports = self.trie.graph.num_outputs(state);
        self.set_num_ports(
            state,
            self.trie.graph.num_inputs(state),
            n_ports + new_transitions.len(),
        );

        // Now shift ports to make space for new ones
        let mut new_offset = self.trie.graph.num_outputs(state);
        for offset in (0..=n_ports).rev() {
            // move existing transition
            let fallback = if offset < n_ports {
                new_offset -= 1;
                let old = self
                    .trie
                    .graph
                    .port_index(state, PortOffset::new_outgoing(offset))
                    .expect("invalid offset");
                let new = self
                    .trie
                    .graph
                    .port_index(state, PortOffset::new_outgoing(new_offset))
                    .expect("invalid offset");
                self.move_out_port(old, new);
                self.trie
                    .graph
                    .port_link(new)
                    .map(|p| self.trie.graph.port_node(p).expect("invalid port"))
            } else {
                None
            };
            // Insert new transitions
            while matches!(new_transitions.last(), Some(&(o, _)) if offset == o) {
                let (_, transition) = new_transitions.pop().expect("just checked");
                new_offset -= 1;
                let new = self
                    .trie
                    .graph
                    .port_index(state, PortOffset::new_outgoing(new_offset))
                    .expect("invalid offset");
                self.trie.weights[new] = Some(transition.clone());
                let next_state = if !self.trie.weights[state].non_deterministic {
                    fallback
                } else {
                    None
                }
                .unwrap_or_else(|| new_state(state, &mut self.trie));
                let in_port = self.add_edge(new, next_state).expect("new port index");
                self.trace[new].0.push(from_world_age.clone());
                self.trace[in_port].0.push(to_world_age.clone());
                next_states.push(next_state);
                used_transitions.push(transition);
            }
            if offset == new_offset {
                // There are no more empty slots. We are done
                break;
            }
        }
        (next_states, used_transitions)
    }

    pub(super) fn set_num_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        // if state == NodeIndex::new(2) {
        // }
        self.trie
            .graph
            .set_num_ports(state, incoming, outgoing, |old, new| {
                let new = new.new_index();
                self.trace.rekey(old, new);
                self.trie.weights.ports.rekey(old, new);
            });
    }

    fn move_out_port(&mut self, old: PortIndex, new: PortIndex) {
        if let Some(in_port) = self.trie.graph.unlink_port(old) {
            self.trie.graph.link_ports(new, in_port).unwrap();
        }
        self.trace.rekey(old, Some(new));
        self.trie.weights.ports.rekey(old, Some(new));
    }
}

pub(super) fn trace_insert<Age: age::Age + Clone + Eq>(
    trace: &mut UnmanagedDenseMap<PortIndex, (Vec<Age>, bool)>,
    from: PortIndex,
    to: PortIndex,
    from_age: Age,
    to_age: Age,
) -> &Age {
    let pos = trace[from].0.iter().position(|x| x == &from_age);
    if let Some(pos) = pos {
        let other = &trace[to].0[pos];
        let merged = to_age.merge(other);
        trace[to].0[pos] = merged;
        &trace[to].0[pos]
    } else {
        trace[from].0.push(from_age);
        trace[to].0.push(to_age);
        trace[to].0.last().expect("just added")
    }
}
