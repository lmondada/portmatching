use std::collections::HashMap;

use portgraph::{PortOffset, Weights};

use crate::addressing::{cache::SpineID, pg::AsPathOffset, SpineAddress};

use super::{base::NodeWeight, BaseGraphTrie};

type S = (SpineID, Vec<PortOffset>, usize);
type SRef<'n> = (SpineID, &'n [PortOffset], usize);

impl SpineAddress for S {
    type AsRef<'n> = SRef<'n>;

    fn as_ref(&self) -> Self::AsRef<'_> {
        (self.0, self.1.as_slice(), self.2)
    }
}

impl<'n> AsPathOffset for SRef<'n> {
    fn as_path_offset(&self) -> (&[PortOffset], usize) {
        (self.1, self.2)
    }
}

impl BaseGraphTrie<S> {
    /// A graph trie enabling caching using [`SpineID`]s.
    ///
    /// This trie is constructed from another [`BaseGraphTrie`]. The addresses
    /// are re-computed and optimised for caching by introducing [`SpineID`]s.
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
            weights[node].address = base.weights[node].address;
            weights[node].non_deterministic = base.weights[node].non_deterministic;
            for port in base.graph.outputs(node) {
                weights[port] = base.weights[port].clone();
            }
        }
        Self {
            graph: base.graph.clone(),
            weights,
            perm_indices: Default::default(),
            trace: Default::default(),
            edge_cnt: Default::default(),
        }
    }
}
