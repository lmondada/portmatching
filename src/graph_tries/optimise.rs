use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt, iter, mem,
};

use bitvec::vec::BitVec;
use itertools::Itertools;
use portgraph::{
    algorithms as pg, Direction, NodeIndex, PortGraph, PortIndex, SecondaryMap, UnmanagedDenseMap,
};

use crate::{
    constraint::ConstraintType,
    graph_tries::trace_insert,
    utils::{causal::is_ancestor, toposort, SetsOfSets},
    Constraint,
};

use super::{root_state, BaseGraphTrie, GraphTrieBuilder, StateID};

// impl<A: Clone + Ord> BaseGraphTrie<C, A> {
impl<C, A> BaseGraphTrie<C, A>
where
    C: ConstraintType + Clone + Ord,
    C::CT: Ord,
    A: Clone + Ord,
    C: fmt::Display,
    A: fmt::Debug,
{
    /// Turn nodes into multiple ones by only relying on elementary constraints
    pub fn optimise<F>(&mut self, mut clone_state: F, cutoff: usize)
    where
        F: FnMut(StateID, StateID),
    {
        let nodes = self
            .graph
            .nodes_iter()
            .filter(|&n| self.graph.num_outputs(n) > cutoff)
            .filter(|&n| self.weight(n).non_deterministic)
            .collect::<Vec<_>>();

        for node in nodes {
            self.split_into_tree(node, &mut clone_state);
        }

        let mut node_copies = SetsOfSets::new();
        // Now turn non-deterministic states into deterministic if number of
        // transitions remains unchanged
        while let Some(node) = {
            // In these cases it's easy (and sensible) to turn non-deterministic
            // states into deterministic
            self.graph.nodes_iter().find(|&n| {
                self.graph.num_outputs(n) > cutoff && self.weight(n).non_deterministic && {
                    let constraints = self
                        .graph
                        .outputs(n)
                        .map(|p| self.weights[p].as_ref())
                        .collect();
                    totally_ordered(&constraints) || mutually_exclusive(&constraints)
                }
            })
        } {
            self.make_deterministic(node, &mut clone_state, &mut node_copies);
        }
    }

    /// Only do this for non-det!
    fn split_into_tree<F: FnMut(StateID, StateID)>(&mut self, node: StateID, mut clone_state: F) {
        let transitions = self
            .graph
            .outputs(node)
            .filter(|&p| self.weights[p].is_some())
            .map(|p| {
                let in_port = self.graph.unlink_port(p).expect("unlinked transition");
                // note: we are leaving dangling inputs, but we know they will
                // be recycled by add_edge below
                self.graph.port_node(in_port).expect("invalid port")
            })
            .collect::<Vec<_>>();
        let fail_transition = self
            .graph
            .outputs(node)
            .find(|&p| self.weights[p].is_none())
            .map(|p| {
                let in_port = self.graph.unlink_port(p).expect("unlinked transition");
                self.graph.port_node(in_port).expect("invalid port")
            });
        let fail_node_weight = fail_transition.map(|f| self.weight(f).clone());
        let constraints = self
            .graph
            .outputs(node)
            .filter_map(|p| self.weights[p].take())
            .map(|c| c.to_elementary())
            .collect::<Vec<_>>();
        let all_constraint_types = constraints
            .iter()
            .flatten()
            .map(|c| c.constraint_type())
            .collect::<BTreeSet<_>>();
        // We space out constraints so they all follow the same layout
        let constraints = constraints.into_iter().map(|c| {
            let mut c = c.into_iter().peekable();
            let mut ret = Vec::new();
            for ct in all_constraint_types.iter() {
                let Some(next) = c.peek() else { break };
                if ct == &next.constraint_type() {
                    ret.push(c.next());
                } else {
                    ret.push(None);
                }
            }
            ret
        });
        self.set_num_ports(node, self.graph.num_inputs(node), 0);
        // Use within loop, allocate once
        let mut new_states = Vec::new();
        for (all_cons, in_node) in constraints.into_iter().zip(transitions) {
            let mut trie = mem::take(self).as_builder();
            let mut states = vec![node];
            for (cons, is_first, is_last) in mark_first_last(&all_cons) {
                new_states.clear();
                for &state in &states {
                    if let Some(cons) = cons {
                        let mut new_state = is_last.then_some(in_node);
                        new_states.append(&mut trie.insert_transitions(
                            state,
                            cons.clone(),
                            |_, trie| *new_state.get_or_insert_with(|| trie.add_state(false)),
                            &0,
                            &0,
                        ));
                    } else {
                        assert!(!is_last, "last transition should not be fail");
                        // This only works because we know node is non-det
                        let mut new_state = if is_first
                            && fail_transition.is_some()
                            && (fail_node_weight.as_ref().unwrap().out_port.is_none()
                                || &trie.trie.weights[node] == fail_node_weight.as_ref().unwrap())
                        {
                            fail_transition
                        } else {
                            None
                        };
                        let next_p = trie.follow_fail(
                            state,
                            |_, trie| *new_state.get_or_insert_with(|| trie.add_state(false)),
                            &0,
                            &0,
                        );
                        new_states.push(trie.trie.graph.port_node(next_p).unwrap());
                    }
                }
                states.clear();
                states.append(&mut new_states);
                if !is_last {
                    for &state in &states {
                        trie.trie.weights[state] = trie.trie.weights[node].clone();
                    }
                }
            }
            let (trie, _) = trie.finalize(root_state(), &mut clone_state);
            *self = trie;
        }
        if let Some(fail_transition) = fail_transition {
            let mut trie = mem::take(self).as_builder();
            let in_p = trie.follow_fail(node, |_, _| fail_transition, &0, &0);
            let mut node = trie.trie.graph.port_node(in_p).expect("invalid port");
            if let Some(out_port) = fail_node_weight.unwrap().out_port {
                let next =
                    trie.valid_start_states(&out_port, node, true, |_, _| fail_transition, &0, &0);
                assert_eq!(next.len(), 1);
                node = next[0];
            }
            *self = trie.skip_finalize();
            debug_assert_eq!(node, fail_transition);
        }
    }

    /// Turn a non-deterministic state into a deterministic one
    ///
    /// Assumes all transitions are either totally ordered or mutually exclusive.
    /// Otherwise calling this is undefined behaviour.
    fn make_deterministic<F>(
        &mut self,
        state: StateID,
        mut clone_state: F,
        node_copies: &mut SetsOfSets<NodeIndex>,
    ) where
        F: FnMut(StateID, StateID),
    {
        if self.graph.num_outputs(state) < 2 {
            // nothing to do
            self.weights[state].non_deterministic = false;
            return;
        }
        let first_p = self.graph.output(state, 0).expect("num_outputs > 0");
        let snd_p = self.graph.output(state, 1).expect("num_outputs > 1");
        if let (Some(first), Some(snd)) =
            (self.weights[first_p].as_ref(), self.weights[snd_p].as_ref())
        {
            if first.and(snd).is_none() {
                // By assumption all transitions are mutually exclusive, so nothing to do
                self.weights[state].non_deterministic = false;
                return;
            }
        }
        for i in (0..self.graph.num_outputs(state)).rev() {
            let p = self.graph.output(state, i).expect("0 <= i < num_outputs");
            let Some(next_p) = self.graph.output(state, i+1) else { continue };
            // By assumption all transitions are totally ordered, so we can merge
            // next_p into p
            self.merge_into(p, next_p, &mut clone_state, node_copies);
        }
        self.weights[state].non_deterministic = false;
    }

    fn merge_into<F>(
        &mut self,
        into: PortIndex,
        other: PortIndex,
        mut clone_state: F,
        node_copies: &mut SetsOfSets<NodeIndex>,
    ) where
        F: FnMut(StateID, StateID),
    {
        let root = self.graph.port_node(other).expect("invalid port");

        let into_link = self.graph.port_link(into).expect("unlinked transition");
        let other_link = self.graph.port_link(other).expect("unlinked transition");

        let next_into = self.graph.port_node(into_link).expect("invalid port");
        let next_other = self.graph.port_node(other_link).expect("invalid port");

        if next_into == next_other {
            return;
        }

        let ages = get_nodes_order(&self.graph, next_other);
        let ages_inv = ages
            .iter()
            .map(|(k, &v)| (v, k))
            .collect::<BTreeMap<_, _>>();
        let get_age = |n| BTreeSet::from_iter([ages[n]]);

        let mut builder = mem::take(self).as_builder();
        builder.trace[other].0.push(get_age(root));
        builder.trace[into].0.push(get_age(root));
        builder.trace[other_link].0.push(get_age(next_other));
        builder.trace[into_link].0.push(get_age(next_other));

        let mut unmerged: VecDeque<_> = [(next_into, ages[next_other])].into();
        let mut clones = BTreeMap::<_, BTreeSet<_>>::new();
        let mut loop_cnt = 0;
        let mut visited = BTreeSet::new();
        while let Some((into, world_age)) = unmerged.pop_front() {
            if visited.contains(&(into, world_age)) {
                continue;
            }
            let model = ages_inv[&world_age];
            loop_cnt += 1;
            if loop_cnt > 1000 {
                panic!("inf loop in merge_into");
            }
            clones.entry(into).or_default().insert(model);
            let out_port = builder.trie.weights[model].out_port.clone();
            let start_states = if let Some(out_port) = out_port {
                let nodes_order = get_nodes_order(&builder.trie.graph, root_state());
                builder.valid_start_states(
                    &out_port,
                    into,
                    true,
                    |from, trie| {
                        let mut all_models = node_copies.get(&model).iter().copied();
                        let to = all_models
                            .find(|&to| !is_ancestor(to, from, &trie.graph, &nodes_order));
                        to.unwrap_or_else(|| {
                            let to = trie.add_state(false);
                            node_copies.insert(&model, to);
                            to
                        })
                    },
                    &BTreeSet::from_iter([world_age]),
                    &get_age(model),
                )
            } else {
                vec![into]
            };
            for into in start_states {
                if !visited.insert((into, world_age)) {
                    continue;
                }
                if builder.trie.weights[model].non_deterministic {
                    // Try to convert to non-determinstic, makes merging easier
                    builder.trie.into_non_deterministic(into);
                }
                unmerged.extend(
                    match (
                        builder.trie.weights[into].non_deterministic,
                        builder.trie.weights[model].non_deterministic,
                    ) {
                        (_, true) => builder.merge_non_det(into, model, &ages, node_copies),
                        (true, false) => builder.merge_non_det_det(into, model, &ages, node_copies),
                        (false, false) => builder.merge_det(into, model, &ages, node_copies),
                    },
                );
            }
        }

        // sanitise before `finalize`
        for (state, &age) in ages.iter() {
            if age != 0 {
                // Does not seem to help
                for &state in node_copies.get(&state).iter() {
                    if builder
                        .trie
                        .graph
                        .inputs(state)
                        .all(|p| builder.trace[p].0.is_empty())
                    {
                        continue;
                    }
                    for p in builder.trie.graph.inputs(state) {
                        if builder.trace[p].0.is_empty() {
                            builder.trace[p].0 = vec![BTreeSet::from_iter([age])];
                        }
                    }
                }
            }
        }
        for state in
            pg::toposort::<BitVec>(&builder.trie.graph, [root_state()], Direction::Outgoing)
        {
            let valid_ages = builder
                .trie
                .graph
                .inputs(state)
                .flat_map(|p| builder.trace[p].0.iter())
                .cloned()
                .collect::<BTreeSet<_>>();
            if valid_ages.is_empty() {
                continue;
            }
            for p in builder.trie.graph.outputs(state) {
                let next_p = builder.trie.graph.port_link(p).expect("invalid link");
                let [t1, t2] = builder
                    .trace
                    .get_disjoint_mut([p, next_p])
                    .expect("link to itself");
                sanitize_ages(&mut t1.0, &mut t2.0, &valid_ages);
            }
        }
        let (trie, new_nodes) = builder.finalize(root, |old, new| {
            for states in node_copies.iter_mut() {
                if states.contains(&old) {
                    states.insert(new);
                }
            }
            clone_state(old, new)
        });
        for (into, new_nodes) in new_nodes {
            for (new, age) in new_nodes.into_iter() {
                let Some(age) = age else { continue };
                for &old in clones.get(&into).into_iter().flatten() {
                    if !age.contains(&ages[old]) {
                        continue;
                    }
                    clone_state(old, new);
                }
            }
        }
        *self = trie;
    }
}

fn sanitize_ages(
    out_ages: &mut Vec<BTreeSet<usize>>,
    in_ages: &mut Vec<BTreeSet<usize>>,
    valid_ages: &BTreeSet<BTreeSet<usize>>,
) {
    let new_out_ages = out_ages
        .iter()
        .flat_map(|age| valid_ages.iter().filter(|v| v.is_superset(age)).cloned())
        .unique()
        .collect_vec();
    let new_in_ages = new_out_ages
        .iter()
        .map(|age| {
            let mut new_in = BTreeSet::<usize>::new();
            for (old_out_age, old_in_age) in out_ages.iter().zip(in_ages.iter()) {
                if old_out_age.is_subset(age) {
                    new_in.extend(old_in_age);
                }
            }
            new_in
        })
        .collect_vec();
    *out_ages = new_out_ages;
    *in_ages = new_in_ages;
}

fn get_nodes_order(g: &PortGraph, root: NodeIndex) -> UnmanagedDenseMap<NodeIndex, usize> {
    let mut nodes_order = UnmanagedDenseMap::new();
    // start counting at 1, 0 means "undef"
    nodes_order[root] = 1;
    for (i, n) in toposort(g, root).enumerate() {
        nodes_order[n] = i + 2;
    }
    nodes_order
}

type Age = BTreeSet<usize>;

impl<C, A> GraphTrieBuilder<C, A, Age>
where
    C: ConstraintType + Clone + Ord,
    C::CT: Ord,
    A: Clone + Ord,
    C: fmt::Display,
    A: fmt::Debug,
{
    fn merge_non_det(
        &mut self,
        into: StateID,
        other: StateID,
        ages: &UnmanagedDenseMap<StateID, usize>,
        node_copies: &mut SetsOfSets<NodeIndex>,
    ) -> Vec<(StateID, usize)> {
        if other == into {
            return vec![];
        }
        let nodes_order = get_nodes_order(&self.trie.graph, root_state());
        let mut unmerged = Vec::new();
        let world_age = BTreeSet::from_iter([ages[other]]);
        for o in 0..self.trie.graph.num_outputs(other) {
            let p = self.trie.graph.output(other, o).expect("invalid port");
            let next_p = self.trie.graph.port_link(p).expect("unlinked transition");
            let next_other = self.trie.graph.port_node(next_p).expect("invalid port");
            let next_age = ages[next_other];
            if ages[next_other] == 0 || self.trace[p].1 {
                // This edge was recently added, no need to copy it
                continue;
            }
            // Mark the trace
            trace_insert(
                &mut self.trace,
                p,
                next_p,
                world_age.clone(),
                BTreeSet::from_iter([next_age]),
            );
            if let Some(cons) = self.trie.weights[p].clone() {
                let next_states = self.insert_transitions(
                    into,
                    cons.clone(),
                    |from, trie| {
                        let mut all_models = node_copies.get(&next_other).iter().copied();
                        let to = all_models.find(|&to| {
                            from != to && !is_ancestor(to, from, &trie.graph, &nodes_order)
                        });
                        to.unwrap_or_else(|| {
                            let to = trie.add_state(false);
                            node_copies.insert(&next_other, to);
                            to
                        })
                    },
                    &world_age,
                    &BTreeSet::from_iter([next_age]),
                );
                for next_into in next_states {
                    if next_into != next_other {
                        unmerged.push((next_into, next_age));
                    }
                }
            } else if self.trie.weights[into].non_deterministic {
                let into_p = self.follow_fail(
                    into,
                    |from, trie| {
                        let mut all_models = node_copies.get(&next_other).iter().copied();
                        let to = all_models.find(|&to| {
                            from != to && !is_ancestor(to, from, &trie.graph, &nodes_order)
                        });
                        to.unwrap_or_else(|| {
                            let to = trie.add_state(false);
                            node_copies.insert(&next_other, to);
                            to
                        })
                    },
                    &world_age,
                    &BTreeSet::from_iter([next_age]),
                );
                let next_into = self.trie.graph.port_node(into_p).expect("invalid port");
                if next_into != next_other {
                    unmerged.push((next_into, next_age));
                }
            } else {
                for o in 0..self.trie.graph.num_outputs(into) {
                    let p = self.trie.graph.output(into, o).expect("invalid port");
                    let next_p = self.trie.graph.port_link(p).expect("unlinked transition");
                    let next_into = self.trie.graph.port_node(next_p).expect("invalid port");
                    let next_age = ages[next_other];
                    trace_insert(
                        &mut self.trace,
                        p,
                        next_p,
                        world_age.clone(),
                        BTreeSet::from_iter([next_age]),
                    );
                    if next_into != next_other {
                        unmerged.push((next_into, next_age));
                    }
                }
            }
        }
        unmerged
    }

    fn merge_non_det_det(
        &mut self,
        into: StateID,
        other: StateID,
        ages: &UnmanagedDenseMap<StateID, usize>,
        node_copies: &mut SetsOfSets<NodeIndex>,
    ) -> Vec<(StateID, usize)> {
        if other == into {
            return vec![];
        }
        let mut unmerged = Vec::new();
        let world_age = ages[other];
        let nodes_order = get_nodes_order(&self.trie.graph, root_state());
        if self.trie.graph.num_outputs(other) > 0 {
            let into_p = self.follow_fail(
                into,
                |from, trie| {
                    let mut all_models = node_copies.get(&other).iter().copied();
                    let to = all_models.find(|&to| {
                        from != to && !is_ancestor(to, from, &trie.graph, &nodes_order)
                    });
                    to.unwrap_or_else(|| {
                        let to = trie.add_state(false);
                        node_copies.insert(&other, to);
                        to
                    })
                },
                &BTreeSet::from_iter([world_age]),
                &BTreeSet::from_iter([world_age]),
            );
            let next_into = self.trie.graph.port_node(into_p).expect("invalid port");
            if next_into != other {
                unmerged.push((next_into, ages[other]));
            }
        }
        unmerged
    }

    fn merge_det(
        &mut self,
        into: StateID,
        other: StateID,
        ages: &UnmanagedDenseMap<StateID, usize>,
        node_copies: &mut SetsOfSets<NodeIndex>,
    ) -> Vec<(StateID, usize)> {
        if into == other {
            return vec![];
        }
        let mut unmerged = Vec::new();
        let mut used_constraints = BTreeSet::new();
        let nodes_order = get_nodes_order(&self.trie.graph, root_state());
        let world_age = ages[other];
        for o in 0..self.trie.graph.num_outputs(other) {
            let p = self.trie.graph.output(other, o).expect("invalid port");
            let next_p = self.trie.graph.port_link(p).expect("unlinked transition");
            let next_other = self.trie.graph.port_node(next_p).expect("invalid port");
            let next_age = ages[next_other];
            if ages[next_other] == 0 || self.trace[p].1 {
                // This edge was recently added, no need to copy it
                continue;
            }
            // Mark the trace
            trace_insert(
                &mut self.trace,
                p,
                next_p,
                BTreeSet::from_iter([world_age]),
                BTreeSet::from_iter([next_age]),
            );
            if let Some(cons) = self.trie.weights[p].clone() {
                let (next_states, new_used) = self.insert_transitions_filtered(
                    into,
                    cons.clone(),
                    |from, trie| {
                        let mut all_models = node_copies.get(&next_other).iter().copied();
                        let to = all_models.find(|&to| {
                            from != to && !is_ancestor(to, from, &trie.graph, &nodes_order)
                        });
                        to.unwrap_or_else(|| {
                            let to = trie.add_state(false);
                            node_copies.insert(&next_other, to);
                            to
                        })
                    },
                    |t| !used_constraints.contains(t),
                    &BTreeSet::from_iter([world_age]),
                    &BTreeSet::from_iter([next_age]),
                );
                used_constraints.extend(new_used);
                for next_into in next_states {
                    if next_into != next_other {
                        unmerged.push((next_into, next_age));
                    }
                }
            } else {
                for transition in self.trie.graph.outputs(into) {
                    if self.trie.weights[transition].is_some()
                        && used_constraints
                            .contains(self.trie.weights[transition].as_ref().unwrap())
                    {
                        continue;
                    }
                    let next_into = self
                        .trie
                        .graph
                        .port_link(transition)
                        .expect("invalid transition");
                    trace_insert(
                        &mut self.trace,
                        transition,
                        next_into,
                        BTreeSet::from_iter([world_age]),
                        BTreeSet::from_iter([next_age]),
                    );
                    let next_into = self.trie.graph.port_node(next_into).expect("invalid port");
                    if next_into != next_other {
                        unmerged.push((next_into, next_age));
                    }
                }
                let last_port = self.trie.graph.outputs(into).last();
                if last_port.is_none() || self.trie.weights[last_port.unwrap()].is_some() {
                    self.follow_fail(
                        into,
                        |from, trie| {
                            let mut all_models = node_copies.get(&next_other).iter().copied();
                            let to = all_models.find(|&to| {
                                from != to && !is_ancestor(to, from, &trie.graph, &nodes_order)
                            });
                            to.unwrap_or_else(|| {
                                let to = trie.add_state(false);
                                node_copies.insert(&next_other, to);
                                to
                            })
                        },
                        &BTreeSet::from_iter([world_age]),
                        &BTreeSet::from_iter([ages[next_other]]),
                    );
                }
            }
        }
        unmerged
    }
}

/// Whether constraints are in a total order, from largest to smallest
fn totally_ordered<C>(constraints: &Vec<Option<&C>>) -> bool
where
    C: Constraint + Ord,
{
    (0..(constraints.len() - 1)).all(|i| {
        let Some(d) = constraints[i + 1] else { return true };
        let Some(c) = constraints[i] else { return false };
        let Some(candd) = c.and(d) else { return false };
        &candd == c
    })
}

/// Whether no two constraints are compatible
fn mutually_exclusive<C>(constraints: &Vec<Option<&C>>) -> bool
where
    C: Constraint + Ord,
{
    for c1 in constraints {
        for c2 in constraints {
            let (Some(c1), Some(c2)) = (c1, c2) else { return false };
            if c1.and(c2).is_some() {
                return false;
            }
        }
    }
    true
}

pub(super) fn get_next_world_age<'a, Map, Age>(
    from: PortIndex,
    to: PortIndex,
    trace: &'a Map,
    world_age: &Age,
) -> &'a Age
where
    Map: SecondaryMap<PortIndex, (Vec<Age>, bool)>,
    Age: Eq,
{
    let pos = trace
        .get(from)
        .0
        .iter()
        .position(|x| x == world_age)
        .unwrap();
    &trace.get(to).0[pos]
}

fn mark_first_last<I: IntoIterator>(all_cons: I) -> impl Iterator<Item = (I::Item, bool, bool)> {
    let mut all_cons = all_cons.into_iter().peekable();
    let mut first_flag = true;
    iter::from_fn(move || {
        all_cons.next().map(|cons| {
            let last = all_cons.peek().is_none();
            let first = first_flag;
            first_flag = false;
            (cons, first, last)
        })
    })
}

#[cfg(test)]
mod tests {
    use portgraph::{PortGraph, Weights};

    use crate::{
        constraint::{Address, NodeRange, PortLabel, SpineAddress, UnweightedAdjConstraint},
        graph_tries::{base::NodeWeight, BaseGraphTrie},
    };

    #[test]
    fn test_optimise() {
        let mut g = PortGraph::new();
        let n_transitions = 4;
        let n0 = g.add_node(0, n_transitions);
        let in_nodes = (0..n_transitions)
            .map(|_| g.add_node(1, 0))
            .collect::<Vec<_>>();
        let ports = g.outputs(n0).collect::<Vec<_>>();
        for (&o, &n) in ports.iter().zip(&in_nodes) {
            let i = g.input(n, 0).unwrap();
            g.link_ports(o, i).unwrap();
        }
        let mut weights = Weights::<NodeWeight<()>, _>::new();
        let spine0 = SpineAddress::new([], 0);
        let spine1 = SpineAddress::new([], 1);
        let spine2 = SpineAddress::new([], 2);
        weights[n0].non_deterministic = true;
        weights[n0].out_port = ().into();
        weights[ports[0]] = Some(UnweightedAdjConstraint::link(
            Address::new(spine0.clone(), 4, PortLabel::Outgoing(0)),
            [
                NodeRange::new(spine0.clone(), -2..=3),
                NodeRange::new(spine1.clone(), -2..=3),
            ],
        ));
        weights[ports[1]] = Some(UnweightedAdjConstraint::link(
            Address::new(spine0.clone(), 4, PortLabel::Outgoing(0)),
            vec![NodeRange::new(spine0.clone(), -2..=2)],
        ));
        weights[ports[2]] = Some(UnweightedAdjConstraint::link(
            Address::new(spine0, 4, PortLabel::Outgoing(0)),
            vec![NodeRange::new(spine1, -2..=2)],
        ));
        weights[ports[3]] = Some(UnweightedAdjConstraint::link(
            Address::new(spine2.clone(), 2, PortLabel::Outgoing(0)),
            vec![NodeRange::new(spine2, -2..=1)],
        ));

        let mut trie = BaseGraphTrie { graph: g, weights };

        trie.optimise(|_, _| {}, 1);
        assert_eq!(
            trie._dotstring(),
            r#"digraph {
0 [shape=plain label=<<table border="1"><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([PortLabel(out(0))])</td></tr></table>>]
0:out0 -> 5:in0 [style=""]
1 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(-2..=2 for ([], 1))])</td></tr></table>>]
1:out0 -> 3:in2 [style=""]
2 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td><td port="in1" align="text" colspan="1" cellpadding="1">1</td></tr><tr><td align="text" border="0" colspan="2">[()]</td></tr><tr><td port="out0" align="text" colspan="2" cellpadding="1">0: Vec([NoMatch(-2..=2 for ([], 1))])</td></tr></table>>]
2:out0 -> 3:in1 [style=""]
3 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td><td port="in1" align="text" colspan="1" cellpadding="1">1</td><td port="in2" align="text" colspan="1" cellpadding="1">2</td></tr><tr><td align="text" border="0" colspan="3"></td></tr></table>>]
4 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
5 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([Match(([], 2): 2))])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([Match(([], 0): 4))])</td></tr></table>>]
5:out0 -> 9:in0 [style=""]
5:out1 -> 6:in0 [style=""]
6 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="3" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="3">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(-2..=3 for ([], 0))])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([NoMatch(-2..=2 for ([], 0))])</td><td port="out2" align="text" colspan="1" cellpadding="1">2: FAIL</td></tr></table>>]
6:out0 -> 7:in0 [style=""]
6:out1 -> 2:in0 [style=""]
6:out2 -> 8:in0 [style=""]
7 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(-2..=3 for ([], 1))])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: FAIL</td></tr></table>>]
7:out0 -> 1:in0 [style=""]
7:out1 -> 2:in1 [style=""]
8 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(-2..=2 for ([], 1))])</td></tr></table>>]
8:out0 -> 3:in0 [style=""]
9 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
9:out0 -> 10:in0 [style=""]
10 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
10:out0 -> 11:in0 [style=""]
11 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(-2..=1 for ([], 2))])</td></tr></table>>]
11:out0 -> 4:in0 [style=""]
}
"#
        );
    }
}
