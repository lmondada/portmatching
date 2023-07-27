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
) -> impl for<'a> Fn(NodeIndex, &'a UnweightedEdge) -> Option<NodeIndex> {
    move |src, &(src_port, tgt_port)| {
        let src_port_index = graph.port_index(src, src_port)?;
        let tgt_port_index = graph.port_link(src_port_index)?;
        let tgt = graph.port_node(tgt_port_index)?;
        if graph.port_offset(tgt_port_index)? != tgt_port {
            return None;
        }
        Some(tgt)
    }
}

/// Check if an edge `e` is valid in a weighted portgraph `g`.
pub(crate) fn validate_weighted_node<'m, G, W, PNode>(
    graph: WeightedGraphRef<G, &'m W>,
) -> impl for<'a> Fn(NodeIndex, &PNode) -> bool + 'm
where
    W: SecondaryMap<NodeIndex, PNode>,
    PNode: Eq,
{
    let (_, weights) = graph.into();
    move |node, node_prop| weights.get(node) == node_prop
}

pub(crate) fn always_true<'a, A, B>(_: A, _: &'a B) -> bool {
    true
}
