use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt, iter,
};

use portgraph::{PortIndex, SecondaryMap};

use crate::{constraint::ConstraintType, utils::toposort, Constraint};

use super::{root_state, BaseGraphTrie, StateID};

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
            self.make_deterministic(node, &mut clone_state);
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
        let mut fail_transition = self
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
            let mut states = vec![node];
            for (cons, is_first, is_last) in mark_first_last(&all_cons) {
                new_states.clear();
                for &state in &states {
                    if let Some(cons) = cons {
                        let mut new_state = is_last.then_some(in_node);
                        new_states.append(&mut self.insert_transitions(
                            state,
                            cons.clone(),
                            &mut new_state,
                            0,
                            0,
                        ));
                    } else {
                        assert!(!is_last, "last transition should not be fail");
                        // This only works because we know node is non-det
                        let mut new_state = if is_first
                            && fail_transition.is_some()
                            && (fail_node_weight.as_ref().unwrap().out_port.is_none()
                                || &self.weights[node] == fail_node_weight.as_ref().unwrap())
                        {
                            fail_transition
                        } else {
                            None
                        };
                        let next_p = self.follow_fail(state, &mut new_state, 0, 0);
                        new_states.push(self.graph.port_node(next_p).unwrap());
                    }
                }
                states.clear();
                states.append(&mut new_states);
                if !is_last {
                    for &state in &states {
                        self.weights[state] = self.weights[node].clone();
                    }
                }
            }
            self.finalize(root_state(), &mut clone_state);
        }
        if fail_transition.is_some() {
            let in_p = self.follow_fail(node, &mut fail_transition, 0, 0);
            let mut node = self.graph.port_node(in_p).expect("invalid port");
            if let Some(out_port) = fail_node_weight.unwrap().out_port {
                let next = self.valid_start_states(&out_port, node, false, &mut fail_transition, 0);
                assert_eq!(next.len(), 1);
                node = next[0];
            }
            debug_assert_eq!(node, fail_transition.unwrap());
        }
    }

    /// Turn a non-deterministic state into a deterministic one
    ///
    /// Assumes all transitions are either totally ordered or mutually exclusive.
    /// Otherwise calling this is undefined behaviour.
    fn make_deterministic<F>(&mut self, state: StateID, mut clone_state: F)
    where
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
            self.merge_into(p, next_p, &mut clone_state);
        }
        self.weights[state].non_deterministic = false;
    }

    fn merge_into<F>(&mut self, into: PortIndex, other: PortIndex, mut clone_state: F)
    where
        F: FnMut(StateID, StateID),
    {
        self.trace = Default::default();
        let root = self.graph.port_node(other).expect("invalid port");
        let ages: BTreeMap<_, _> = toposort(&self.graph, root).zip(0..).collect();

        self.trace[other].0.push(ages[&root]);
        self.trace[into].0.push(ages[&root]);

        let into = self.graph.port_link(into).expect("unlinked transition");
        let into_node = self.graph.port_node(into).expect("invalid port");
        let other = self.graph.port_link(other).expect("unlinked transition");
        let other_node = self.graph.port_node(other).expect("invalid port");

        if into_node == other_node {
            return;
        }

        self.trace[other].0.push(ages[&other_node]);
        self.trace[into].0.push(ages[&other_node]);

        let mut unmerged: VecDeque<_> = [(into_node, other_node, ages[&other_node])].into();
        let mut clones = BTreeMap::<_, BTreeSet<_>>::new();
        let mut all_ogs = BTreeSet::new();
        while let Some((into, other, from_world_age)) = unmerged.pop_front() {
            clones.entry(into).or_default().insert(other);
            all_ogs.insert(other);
            let out_port = self.weights[other].out_port.clone();
            let start_states = if let Some(out_port) = out_port {
                self.valid_start_states(&out_port, into, true, &mut Some(other), from_world_age)
            } else {
                vec![into]
            };
            for into in start_states {
                unmerged.extend(
                    match (
                        self.weights[into].non_deterministic,
                        self.weights[other].non_deterministic,
                    ) {
                        (_, true) => self.merge_non_det(into, other, from_world_age, &ages),
                        (true, false) => self.merge_non_det_det(into, other, from_world_age),
                        (false, false) => self.merge_det(into, other, from_world_age, &ages),
                    },
                );
            }
        }

        // little trick to avoid breaking up OG state in `finalize`
        for &og in &all_ogs {
            for p in self.graph.inputs(og) {
                let v = &mut self.trace[p].0;
                if v.is_empty() {
                    v.push(ages[&og]);
                }
                debug_assert_eq!(*v, vec![ages[&og]]);
            }
        }
        // let mut weights = Weights::<String, _>::new();
        // for p in self.graph.ports_iter() {
        //     weights[p] = format!(
        //         "{} [{:?}]",
        //         self.weights[p]
        //             .as_ref()
        //             .map(|c| c.to_string())
        //             .unwrap_or_default(),
        //         self.trace[p]
        //     );
        // }
        // fs::write(format!("trie_mid_{into:?}.gv"), dot_string_weighted(&self.graph, &weights)).unwrap();

        for new in self.finalize(root, |old, new| {
            if let Some(old) = clones.get(&old).cloned() {
                clones.entry(new).or_default().extend(old);
            }
            clone_state(old, new)
        }) {
            for &old in clones.get(&new).into_iter().flatten() {
                clone_state(old, new);
            }
        }
    }

    fn merge_non_det(
        &mut self,
        into: StateID,
        other: StateID,
        from_world_age: usize,
        ages: &BTreeMap<StateID, usize>,
    ) -> Vec<(StateID, StateID, usize)> {
        if other == into {
            return vec![];
        }
        let mut unmerged = Vec::new();
        for o in 0..self.graph.num_outputs(other) {
            let p = self.graph.output(other, o).expect("invalid port");
            let next_p = self.graph.port_link(p).expect("unlinked transition");
            let next_other = self.graph.port_node(next_p).expect("invalid port");
            // Mark the trace
            if !self.trace[p].0.contains(&from_world_age) {
                self.trace[p].0.push(from_world_age);
                self.trace[next_p].0.push(ages[&next_other]);
            }
            if let Some(cons) = self.weights[p].clone() {
                let (next_states, ages) = self.insert_transitions_ages(
                    into,
                    cons.clone(),
                    &mut Some(next_other),
                    from_world_age,
                    ages[&next_other],
                );
                for (next_into, to_world_age) in next_states.into_iter().zip(ages) {
                    if next_into != next_other {
                        unmerged.push((next_into, next_other, to_world_age));
                    }
                }
            } else if self.weights[into].non_deterministic {
                let into_p = self.follow_fail(
                    into,
                    &mut Some(next_other),
                    from_world_age,
                    ages[&next_other],
                );
                let from = self.graph.port_link(into_p).expect("unlinked transition");
                let to_world_age = get_next_world_age(from, into_p, &self.trace, from_world_age);
                let next_into = self.graph.port_node(into_p).expect("invalid port");
                if next_into != next_other {
                    unmerged.push((next_into, next_other, to_world_age));
                }
            } else {
                // Delay world age
                if into != next_other {
                    unmerged.push((into, next_other, from_world_age));
                }
            }
        }
        unmerged
    }

    fn merge_non_det_det(
        &mut self,
        into: StateID,
        other: StateID,
        from_world_age: usize,
    ) -> Vec<(StateID, StateID, usize)> {
        if other == into {
            return vec![];
        }
        let mut unmerged = Vec::new();
        if self.graph.num_outputs(other) > 0 {
            let into_p = self.follow_fail(into, &mut Some(other), from_world_age, from_world_age);
            let into = self.graph.port_node(into_p).expect("invalid port");
            if into != other {
                unmerged.push((into, other, from_world_age));
            }
        }
        unmerged
    }

    fn merge_det(
        &mut self,
        into: StateID,
        other: StateID,
        from_world_age: usize,
        ages: &BTreeMap<StateID, usize>,
    ) -> Vec<(StateID, StateID, usize)> {
        if into == other {
            return vec![];
        }
        let mut unmerged = Vec::new();
        let mut used_constraints = BTreeSet::new();
        for o in 0..self.graph.num_outputs(other) {
            let p = self.graph.output(other, o).expect("invalid port");
            let next_p = self.graph.port_link(p).expect("unlinked transition");
            let next_other = self.graph.port_node(next_p).expect("invalid port");
            // Mark the trace
            if !self.trace[p].0.contains(&from_world_age) {
                self.trace[p].0.push(from_world_age);
                self.trace[next_p].0.push(ages[&next_other]);
            }
            if let Some(cons) = self.weights[p].clone() {
                let (next_states, new_used, next_ages) = self.insert_transitions_filtered(
                    into,
                    cons.clone(),
                    &mut Some(next_other),
                    |t| !used_constraints.contains(t),
                    from_world_age,
                    ages[&next_other],
                );
                used_constraints.extend(new_used);
                for (next_into, to_world_age) in next_states.into_iter().zip(next_ages) {
                    if next_into != next_other {
                        unmerged.push((next_into, next_other, to_world_age));
                    }
                }
            } else {
                for transition in self.graph.outputs(into) {
                    if self.weights[transition].is_some()
                        && used_constraints.contains(self.weights[transition].as_ref().unwrap())
                    {
                        continue;
                    }
                    let next_into = self
                        .graph
                        .port_link(transition)
                        .expect("invalid transition");
                    if !self.trace[transition].0.contains(&from_world_age) {
                        self.trace[transition].0.push(from_world_age);
                        self.trace[next_into].0.push(ages[&next_other]);
                    }
                    let to_world_age =
                        get_next_world_age(transition, next_into, &self.trace, from_world_age);
                    let next_into = self.graph.port_node(next_into).expect("invalid port");
                    if next_into != next_other {
                        unmerged.push((next_into, next_other, to_world_age));
                    }
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

pub(super) fn get_next_world_age<Map>(
    from: PortIndex,
    to: PortIndex,
    trace: &Map,
    from_world_age: usize,
) -> usize
where
    Map: SecondaryMap<PortIndex, (Vec<usize>, bool)>,
{
    let pos = trace
        .get(from)
        .0
        .iter()
        .position(|&x| x == from_world_age)
        .unwrap();
    trace.get(to).0[pos]
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
    use portgraph::{PortGraph, SecondaryMap, Weights};

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

        let mut trie = BaseGraphTrie {
            graph: g,
            weights,
            // these don't matter
            world_age: 0,
            trace: SecondaryMap::new(),
        };

        trie.optimise(|_, _| {}, 1);
        assert_eq!(
            trie._dotstring(),
            r#"digraph {
0 [shape=plain label=<<table border="1"><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([PortLabel(Outgoing(0))])</td></tr></table>>]
0:out0 -> 5:in0 [style=""]
1 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
1:out0 -> 13:in0 [style=""]
2 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
2:out0 -> 3:in1 [style=""]
3 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td><td port="in1" align="text" colspan="1" cellpadding="1">1</td></tr><tr><td align="text" border="0" colspan="2"></td></tr></table>>]
4 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
5 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 2 }, ind: 2 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 0 }, ind: 4 })])</td></tr></table>>]
5:out0 -> 9:in0 [style=""]
5:out1 -> 6:in0 [style=""]
6 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="3" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="3">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: -2..=3 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: -2..=2 })])</td><td port="out2" align="text" colspan="1" cellpadding="1">2: FAIL</td></tr></table>>]
6:out0 -> 7:in0 [style=""]
6:out1 -> 2:in0 [style=""]
6:out2 -> 8:in0 [style=""]
7 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=3 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: FAIL</td></tr></table>>]
7:out0 -> 1:in0 [style=""]
7:out1 -> 12:in0 [style=""]
8 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
8:out0 -> 3:in0 [style=""]
9 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
9:out0 -> 10:in0 [style=""]
10 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
10:out0 -> 11:in0 [style=""]
11 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 2 }, range: -2..=1 })])</td></tr></table>>]
11:out0 -> 4:in0 [style=""]
12 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
12:out0 -> 13:in1 [style=""]
13 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td><td port="in1" align="text" colspan="1" cellpadding="1">1</td></tr><tr><td align="text" border="0" colspan="2"></td></tr></table>>]
}
"#
        );
    }
}
