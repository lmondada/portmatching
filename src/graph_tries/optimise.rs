use std::{
    collections::{BTreeSet, VecDeque},
    iter,
};

use portgraph::PortIndex;

use crate::{constraint::ConstraintType, Constraint};

use super::{BaseGraphTrie, StateID};

// impl<A: Clone + Ord> BaseGraphTrie<C, A> {
impl<C, A> BaseGraphTrie<C, A>
where
    C: ConstraintType + Clone + Ord,
    C::CT: Ord,
    A: Clone + Ord,
{
    /// Turn nodes into multiple ones by only relying on elementary constraints
    pub fn optimise(&mut self, cutoff: usize) {
        let nodes = self
            .graph
            .nodes_iter()
            .filter(|&n| self.graph.num_outputs(n) > cutoff)
            .collect::<Vec<_>>();

        for node in nodes {
            self.split_into_tree(node);
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
            self.make_deterministic(node);
        }
    }

    fn split_into_tree(&mut self, node: StateID) {
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
        let non_det = self.weight(node).non_deterministic;
        // If non-det we space out constraints so they all follow the same layout
        let constraints = constraints.into_iter().map(|c| {
            let mut c = c.into_iter().peekable();
            if non_det {
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
            } else {
                c.map(Some).collect()
            }
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
                            false,
                        ));
                    } else {
                        assert!(!is_last, "last transition should not be fail");
                        // This only works because we know node is non-det
                        let mut new_state = if is_first
                            && (fail_node_weight.as_ref().unwrap().out_port.is_none()
                                || &self.weights[node] == fail_node_weight.as_ref().unwrap())
                        {
                            fail_transition
                        } else {
                            None
                        };
                        let next_p = self.follow_fail(state, &mut new_state);
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
                // self.edge_cnt += 1;
            }
            self.finalize(|_, _| {});
        }
        if fail_transition.is_some() {
            if let Some(out_port) = fail_node_weight.unwrap().out_port {
                let next = self.valid_start_states(&out_port, node, false, &mut fail_transition);
                debug_assert_eq!(next, vec![fail_transition.unwrap()])
            } else {
                self.follow_fail(node, &mut fail_transition);
            }
        }
    }

    /// Turn a non-deterministic state into a deterministic one
    ///
    /// Assumes all transitions are either totally ordered or mutually exclusive.
    /// Otherwise calling this is undefined behaviour.
    fn make_deterministic(&mut self, state: StateID) {
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
            self.merge_into(p, next_p);
        }
        self.weights[state].non_deterministic = false;
    }

    fn merge_into(&mut self, into: PortIndex, other: PortIndex) {
        let into = self.graph.port_link(into).expect("unlinked transition");
        let other = self.graph.port_link(other).expect("unlinked transition");
        let mut unmerged: VecDeque<_> = [(
            self.graph.port_node(into).expect("invalid port"),
            self.graph.port_node(other).expect("invalid port"),
        )]
        .into();
        while let Some((into, other)) = unmerged.pop_front() {
            for p in self.graph.outputs(other) {
                let next_p = self.graph.port_link(p).expect("unlinked transition");
                let next_other = self.graph.port_node(next_p).expect("invalid port");
                if let Some(cons) = self.weights[p].clone() {
                    let out_port = self.weights[other]
                        .out_port
                        .clone()
                        .expect("A node with transitions must have out_port");
                    let start_states =
                        self.valid_start_states(&out_port, into, true, &mut Some(other));
                    let mut next_states = BTreeSet::new();
                    for state in start_states {
                        next_states.extend(
                            self.insert_transitions(
                                state,
                                cons.clone(),
                                &mut Some(next_other),
                                true,
                            )
                            .into_iter(),
                        );
                    }
                    for next_into in next_states {
                        if next_into != next_other {
                            unmerged.push_back((next_into, next_other));
                        }
                    }
                } else {
                    let into_p = self.follow_fail(into, &mut Some(next_other));
                    let next_into = self.graph.port_node(into_p).expect("invalid port");
                    if next_into != next_other {
                        unmerged.push_back((next_into, next_other));
                    }
                }
            }
        }
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
            Address::new(spine0.clone(), 4, PortLabel::Outgoing(0)),
            vec![NodeRange::new(spine1.clone(), -2..=2)],
        ));
        weights[ports[3]] = Some(UnweightedAdjConstraint::link(
            Address::new(spine2.clone(), 2, PortLabel::Outgoing(0)),
            vec![NodeRange::new(spine2.clone(), -2..=1)],
        ));

        let mut trie = BaseGraphTrie {
            graph: g,
            weights: weights,
            // these don't matter
            edge_cnt: 0,
            trace: SecondaryMap::new(),
        };

        trie.optimise(1);
        assert_eq!(
            trie._dotstring(),
            r#"digraph {
0 [shape=plain label=<<table border="1"><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([PortLabel(Outgoing(0))])</td></tr></table>>]
0:out0 -> 5:in0 [style=""]
1 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
2 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
2:out0 -> 3:in1 [style=""]
3 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td><td port="in1" align="text" colspan="1" cellpadding="1">1</td><td port="in2" align="text" colspan="1" cellpadding="1">2</td></tr><tr><td align="text" border="0" colspan="3"></td></tr></table>>]
4 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
5 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 2 }, ind: 2 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 0 }, ind: 4 })])</td></tr></table>>]
5:out0 -> 9:in0 [style=""]
5:out1 -> 6:in0 [style=""]
6 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="3" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="3">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: -2..=3 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: -2..=2 })])</td><td port="out2" align="text" colspan="1" cellpadding="1">2: FAIL</td></tr></table>>]
6:out0 -> 7:in0 [style=""]
6:out1 -> 2:in0 [style=""]
6:out2 -> 8:in0 [style=""]
7 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2">[()]</td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=3 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
7:out0 -> 1:in0 [style=""]
7:out1 -> 3:in2 [style=""]
8 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: -2..=2 })])</td></tr></table>>]
8:out0 -> 3:in0 [style=""]
9 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
9:out0 -> 10:in0 [style=""]
10 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
10:out0 -> 11:in0 [style=""]
11 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red">[()]</font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 2 }, range: -2..=1 })])</td></tr></table>>]
11:out0 -> 4:in0 [style=""]
}
"#
        );
    }
}
