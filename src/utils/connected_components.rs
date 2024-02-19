use std::collections::BTreeMap;

use petgraph::unionfind::UnionFind;
use portgraph::{LinkView, NodeIndex};

/// Return a partition of the graph vertices into connected components.
pub fn connected_components<G: LinkView>(graph: &G) -> Vec<Vec<NodeIndex>> {
    let Some(max_v) = graph.nodes_iter().max().map(|n| n.index()) else {
        return Vec::new();
    };
    let mut uf = UnionFind::new(max_v + 1);
    for v in graph.nodes_iter() {
        for n in graph.all_neighbours(v) {
            uf.union(v.index(), n.index());
        }
    }
    let labels = uf.into_labeling();
    let mut label_to_ind = BTreeMap::new();
    let mut partition = Vec::new();
    for v in graph.nodes_iter() {
        let l = labels[v.index()];
        let &mut ind = label_to_ind.entry(l).or_insert_with(|| {
            partition.push(Vec::new());
            partition.len() - 1
        });
        partition[ind].push(v);
    }
    partition
}

/// Whether the graph is connected.
pub fn is_connected<G>(graph: &G) -> bool
where
    G: LinkView,
{
    connected_components(graph).len() == 1
}
