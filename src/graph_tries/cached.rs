use std::{
    collections::HashMap,
    fmt::{self, Display}, vec, slice::Iter, iter::Cloned,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset, Weights};

use crate::{
    addressing::{
        cache::{Cache, SpineID},
        Address, Rib, PortGraphAddressing, SkeletonAddressing,
    },
    utils::{follow_path, port_opposite},
};

use super::{BaseGraphTrie, GraphTrie, StateID, StateTransition, base::NodeWeight};

type S = (SpineID, Vec<PortOffset>, usize);

impl BaseGraphTrie<S> {
    //! A graph trie enabling caching using [`SpineID`]s.
    //! 
    //! This trie is constructed from another [`BaseGraphTrie`]. The addresses
    //! are re-computed and optimised for caching by introducing [`SpineID`]s.
    pub fn new(base: &BaseGraphTrie<(Vec<PortOffset>, usize)>) -> Self {
        let mut weights = Weights::new();
        let mut existing_spines = HashMap::new();
        let mut next_ind = 0;
        for node in base.graph.nodes_iter() {
            let weight: &mut NodeWeight<S> = weights.nodes.get_mut(node);
            weight.out_port = base.weight(node).out_port;
            if let Some(spine) = base.weight(node).spine.as_ref() {
                let mut new_spine = Vec::with_capacity(spine.len());
                for s in spine {
                    let &mut spine_id =
                        existing_spines
                            .entry((s.0.clone(), s.1))
                            .or_insert_with(|| {
                                let ret = next_ind;
                                next_ind += 1;
                                SpineID(ret)
                            });
                    new_spine.push((spine_id, s.0.clone(), s.1));
                }
                weights[node].spine = Some(new_spine);
            }
            weights[node].address = base.weights[node].address.clone();
            weights[node].non_deterministic = base.weights[node].non_deterministic;
            for port in base.graph.outputs(node) {
                weights[port] = base.weights[port].clone();
            }
        }
        Self {
            graph: base.graph.clone(),
            weights,
            perm_indices: Default::default()
        }
    }
}