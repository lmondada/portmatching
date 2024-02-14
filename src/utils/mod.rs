//! Utility functions.

#[cfg(test)]
pub(crate) mod test;

mod depth;
mod pre_order;
pub use depth::is_connected;
use portgraph::{LinkView, NodeIndex, SecondaryMap};

use crate::{patterns::UnweightedEdge, WeightedGraphRef};

/// Check if an edge `e` is valid in a portgraph `g` without weights.
pub(crate) fn validate_unweighted_edge<G: LinkView>(
    graph: G,
) -> impl for<'a> Fn(NodeIndex, &'a UnweightedEdge) -> Vec<Option<NodeIndex>> {
    move |src, &(src_port, tgt_port)| {
        let Some(src_port_index) = graph.port_index(src, src_port) else {
            return Vec::new();
        };
        let tgt_port_indices = graph.port_links(src_port_index);
        tgt_port_indices
            .map(|(_, tgt_port_index)| {
                let tgt = graph.port_node(tgt_port_index)?;
                (graph.port_offset(tgt_port_index)? == tgt_port).then_some(tgt)
            })
            .collect()
    }
}

/// Check if an edge `e` is valid in a weighted portgraph `g`.
pub(crate) fn validate_weighted_node<G, W, PNode>(
    graph: WeightedGraphRef<G, &W>,
) -> impl for<'a> Fn(NodeIndex, &PNode) -> bool + '_
where
    W: SecondaryMap<NodeIndex, PNode>,
    PNode: Eq,
{
    let (_, weights) = graph.into();
    move |node, node_prop| weights.get(node) == node_prop
}

pub(crate) fn always_true<A, B>(_: A, _: &B) -> bool {
    true
}
