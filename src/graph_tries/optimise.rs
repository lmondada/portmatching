use std::{
    collections::{BTreeSet, HashSet},
    fmt, iter,
};

use bitvec::vec::BitVec;
use itertools::Itertools;
use portgraph::{algorithms as pg, NodeIndex, PortGraph};
use portgraph::{Direction, PortOffset};

use crate::{
    constraint::{mutually_exclusive, totally_ordered, ConstraintType},
    Constraint,
};

use super::{root_state, BaseGraphTrie, StateID};

// impl<A: Clone + Ord> BaseGraphTrie<C, A> {
impl<C, A> BaseGraphTrie<C, A>
where
    C: Constraint + Clone + Ord,
    A: Clone + Ord,
{
    /// The largest non-deterministic states, in topological order
    pub(crate) fn large_non_det_states(&self, cutoff: usize, depth_cutoff: usize) -> Vec<StateID> {
        // Always process parent nodes first when converting to det to avoid
        // node count explosion
        pg::toposort_filtered::<BitVec<_>>(
            &self.graph,
            [root_state()],
            Direction::Outgoing,
            move |n| depth(n, &self.graph) < depth_cutoff,
            |_, _| true,
        )
        .filter(move |&n| self.graph.num_outputs(n) > cutoff)
        .filter(|&n| self.weight(n).non_deterministic)
        .collect()
    }

    pub(crate) fn all_constraints(&self, state: StateID) -> Vec<Option<&C>> {
        self.graph
            .outputs(state)
            .map(|p| self.weights[p].as_ref())
            .collect()
    }

    pub(crate) fn split_into_det_tree(&mut self, node: StateID) -> Vec<StateID>
    where
        A: fmt::Debug,
        C: ConstraintType,
        C::CT: Ord,
    {
        // Split into non-deterministic tree
        let bottom_nodes = self.split_into_tree(node);

        let inner_nodes = pg::toposort_filtered::<HashSet<_>>(
            &self.graph,
            [node],
            Direction::Outgoing,
            |n| !bottom_nodes.contains(&n),
            |_, _| true,
        )
        .collect_vec();

        // Make inner nodes of tree deterministic
        for node in inner_nodes {
            let constraints = self.all_constraints(node);
            if mutually_exclusive(&constraints) {
                self.weights[node].non_deterministic = true;
                continue;
            } else if !totally_ordered(&constraints) {
                // Do not attempt to make deterministic
                continue;
            }
            for i in (0..self.graph.num_outputs(node)).rev() {
                let child_i = |i| {
                    let out = self.graph.output(node, i)?;
                    let link = self.graph.port_link(out).expect("unlinked port");
                    self.graph.port_node(link)
                };
                let Some(right) = child_i(i + 1) else {
                    continue
                };
                let left = child_i(i).expect("0 <= i < num_outputs");
                assert_eq!(self.weights[left], self.weights[right]);
                assert!(self.weights[left].non_deterministic);

                // merge right into left
                let mut new_transitions = self
                    .graph
                    .outputs(right)
                    .map(|right_p| {
                        let mut offset = 0;
                        let right_con = self.weights[right_p].as_ref();
                        while offset < self.graph.num_outputs(left) {
                            let left_p = self.graph.output(left, offset).expect("invalid port");
                            let left_con = self.weights[left_p].as_ref();
                            let Some(right_con) = right_con else {
                                offset += 1;
                                continue
                            };
                            let Some(left_con) = left_con else {
                                break
                            };
                            if right_con.constraint_type() < left_con.constraint_type() {
                                break;
                            }
                            offset += 1;
                            if right_con == left_con {
                                break;
                            }
                        }
                        let next_right = self.graph.port_link(right_p).expect("unlinked port");
                        let next_right = self.graph.port_node(next_right).expect("invalid port");
                        (offset, right_con.cloned(), next_right)
                    })
                    .collect_vec();

                // Create new ports for the new transitions
                let n_ports = self.graph.num_outputs(left);
                self.set_num_ports(
                    left,
                    self.graph.num_inputs(left),
                    n_ports + new_transitions.len(),
                );

                // Now shift ports to make space for new ones
                let mut new_offset = self.graph.num_outputs(left);
                for offset in (0..=n_ports).rev() {
                    // move existing transition
                    if offset < n_ports {
                        new_offset -= 1;
                        let old = self
                            .graph
                            .port_index(left, PortOffset::new_outgoing(offset))
                            .expect("invalid offset");
                        let new = self
                            .graph
                            .port_index(left, PortOffset::new_outgoing(new_offset))
                            .expect("invalid offset");
                        if let Some(in_port) = self.graph.unlink_port(old) {
                            self.graph.link_ports(new, in_port).unwrap();
                        }
                        self.weights.ports.rekey(old, Some(new));
                    }
                    // Insert new transitions
                    while matches!(new_transitions.last(), Some(&(o, _, _)) if offset == o) {
                        let (_, transition, next_state) =
                            new_transitions.pop().expect("just checked");
                        new_offset -= 1;
                        let new = self
                            .graph
                            .port_index(left, PortOffset::new_outgoing(new_offset))
                            .expect("invalid offset");
                        self.weights[new] = transition.clone();
                        self.add_edge(new, next_state).expect("new port index");
                    }
                    if offset == new_offset {
                        // There are no more empty slots. We are done
                        break;
                    }
                }
            }
            self.remove_duplicate_ports(node);
            self.weights[node].non_deterministic = false;
        }
        bottom_nodes
    }

    pub(crate) fn remove_duplicate_ports(&mut self, node: StateID) {
        // Clean up duplicates
        let mut duplicates = BTreeSet::new();
        let mut shift_left = 0;
        for i in 0..self.graph.num_outputs(node) {
            let p = self.graph.output(node, i).expect("0 <= i < num_outputs");
            let c = self.weights[p].clone();
            if !duplicates.insert(c) {
                shift_left += 1;
                let child = self.graph.unlink_port(p).expect("unlinked port");
                self.weights[p] = None;
                // Note: we know this node was added recently, so all descendants
                // will also have other parents and no node will be orphan
                let child = self.graph.port_node(child).expect("unlinked port");
                self.remove_stray_inputs(child);
            } else if shift_left > 0 {
                let new = self
                    .graph
                    .port_index(node, PortOffset::new_outgoing(i - shift_left))
                    .expect("invalid i - shift_left");
                if let Some(in_port) = self.graph.unlink_port(p) {
                    self.graph.link_ports(new, in_port).unwrap();
                }
                self.weights.ports.rekey(p, Some(new));
            }
        }
        self.set_num_ports(
            node,
            self.graph.num_inputs(node),
            self.graph.num_outputs(node) - shift_left,
        );
    }

    /// This only works for non-deterministic states
    fn split_into_tree(&mut self, node: StateID) -> Vec<StateID>
    where
        C: ConstraintType,
        C::CT: Ord,
    {
        assert!(self.weight(node).non_deterministic);
        let transitions = self
            .graph
            .outputs(node)
            .map(|p| {
                let in_port = self.graph.unlink_port(p).expect("unlinked transition");
                // note: we are leaving dangling inputs, but we know they will
                // be recycled by add_edge below
                self.graph.port_node(in_port).expect("invalid port")
            })
            .collect::<Vec<_>>();
        let constraints = self
            .graph
            .outputs(node)
            .map(|p| self.weights[p].take())
            .map(|c| {
                if let Some(c) = c {
                    c.to_elementary()
                } else {
                    vec![]
                }
            })
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
                if let Some(next) = c.peek() {
                    if ct == &next.constraint_type() {
                        ret.push(c.next());
                    } else {
                        ret.push(None);
                    }
                } else {
                    ret.push(None);
                }
            }
            ret
        });

        self.set_num_ports(node, self.graph.num_inputs(node), 0);

        let mut bottom_states = Vec::new();
        for (all_cons, in_node) in constraints.into_iter().zip(transitions) {
            let mut state = node;
            for (cons, _, is_last) in mark_first_last(&all_cons) {
                let mut new_state = is_last.then_some(in_node);
                if is_last {
                    bottom_states.push(state);
                }
                if let Some(cons) = cons {
                    let next_states =
                        self.insert_transitions(state, cons.clone(), &mut new_state, 0, 0);
                    assert_eq!(next_states.len(), 1);
                    state = next_states[0];
                } else {
                    let next_p = self.follow_fail(state, &mut new_state, 0, 0);
                    state = self.graph.port_node(next_p).unwrap();
                }
                if !is_last {
                    self.weights[state] = self.weights[node].clone();
                }
            }
        }
        bottom_states
    }
}

fn depth(n: NodeIndex, graph: &PortGraph) -> usize {
    let mut d = 1;
    let mut layer = BTreeSet::from_iter([n]);
    while !layer.contains(&root_state()) {
        d += 1;
        layer = layer
            .into_iter()
            .flat_map(|n| {
                graph
                    .input_links(n)
                    .flatten()
                    .map(|p| graph.port_node(p).unwrap())
            })
            .collect();
    }
    d
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
