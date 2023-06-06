use std::{collections::BTreeSet, iter, mem};

use crate::constraint::ConstraintType;

use super::{BaseGraphTrie, StateID};

// impl<A: Clone + Ord> BaseGraphTrie<C, A> {
impl<C, A> BaseGraphTrie<C, A>
where
    C: ConstraintType + Clone + Ord,
    C::CT: Ord,
    A: Clone + Ord,
{
    /// Turn nodes into multiple ones by only relying on elementary constraints
    pub fn optimise(&mut self) {
        let cutoff = 2;
        let nodes = self
            .graph
            .nodes_iter()
            .filter(|&n| self.graph.num_outputs(n) > cutoff)
            .collect::<Vec<_>>();

        for node in nodes {
            self.split_into_tree(node);
        }
    }

    fn split_into_tree(&mut self, node: StateID) {
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
            .map(|p| mem::take(&mut self.weights[p]).map(|c| c.to_elementary()))
            .collect::<Vec<_>>();
        let all_constraint_types = constraints
            .iter()
            .flatten()
            .flatten()
            .map(|c| c.constraint_type())
            .collect::<BTreeSet<_>>();
        let non_det = self.weight(node).non_deterministic;
        // If non-det we space out constraints so they all follow the same layout
        let constraints = constraints.into_iter().map(|c| {
            let mut c = c?.into_iter().peekable();
            Some(if non_det {
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
            })
        });
        self.set_num_ports(node, self.graph.num_inputs(node), 0);
        // Use within loop, allocate once
        let mut new_states = Vec::new();
        for (all_cons, in_node) in constraints.into_iter().zip(transitions) {
            if let Some(all_cons) = all_cons.as_ref() {
                let mut states = vec![node];
                for (cons, is_last) in mark_last(all_cons) {
                    let mut new_state = is_last.then_some(in_node);
                    new_states.clear();
                    for &state in &states {
                        if let Some(cons) = cons {
                            new_states.append(&mut self.insert_transitions(
                                state,
                                cons.clone(),
                                &mut new_state,
                                false,
                            ));
                        } else {
                            // This only works because we know node is non-det
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
            } else {
                self.follow_fail(node, &mut Some(in_node));
            }
        }
    }
}

fn mark_last<I: IntoIterator>(all_cons: I) -> impl Iterator<Item = (I::Item, bool)> {
    let mut all_cons = all_cons.into_iter().peekable();
    iter::from_fn(move || {
        all_cons.next().map(|cons| {
            let last = all_cons.peek().is_none();
            (cons, last)
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

        trie.optimise();
        assert_eq!(
            trie._dotstring(),
            r#"digraph {
0 [shape=plain label=<<table border="1"><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([PortLabel(Outgoing(0))])</td></tr></table>>]
0:out0 -> 5:in0 [style=""]
1 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
2 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
3 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
4 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"></td></tr></table>>]
5 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="2" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="2"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 2 }, ind: 2 })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([Match(NodeAddress { spine: SpineAddress { path: [], offset: 0 }, ind: 4 })])</td></tr></table>>]
5:out0 -> 9:in0 [style=""]
5:out1 -> 6:in0 [style=""]
6 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="3" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="3"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: NonEmpty { start: 2, end: 3 } })])</td><td port="out1" align="text" colspan="1" cellpadding="1">1: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 0 }, range: NonEmpty { start: 2, end: 2 } })])</td><td port="out2" align="text" colspan="1" cellpadding="1">2: FAIL</td></tr></table>>]
6:out0 -> 7:in0 [style=""]
6:out1 -> 2:in0 [style=""]
6:out2 -> 8:in0 [style=""]
7 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: NonEmpty { start: 2, end: 3 } })])</td></tr></table>>]
7:out0 -> 1:in0 [style=""]
8 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 1 }, range: NonEmpty { start: 2, end: 2 } })])</td></tr></table>>]
8:out0 -> 3:in0 [style=""]
9 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
9:out0 -> 10:in0 [style=""]
10 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: FAIL</td></tr></table>>]
10:out0 -> 11:in0 [style=""]
11 [shape=plain label=<<table border="1"><tr><td port="in0" align="text" colspan="1" cellpadding="1">0</td></tr><tr><td align="text" border="0" colspan="1"><font color="red"></font></td></tr><tr><td port="out0" align="text" colspan="1" cellpadding="1">0: Vec([NoMatch(NodeRange { spine: SpineAddress { path: [], offset: 2 }, range: NonEmpty { start: 2, end: 1 } })])</td></tr></table>>]
11:out0 -> 4:in0 [style=""]
}
"#
        );
    }
}
