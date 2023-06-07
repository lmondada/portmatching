use std::collections::BTreeSet;

use portgraph::{algorithms as pg, Direction, NodeIndex, PortGraph};

/// Topological sorting of the graph
///
/// Unlike `portgraph::algorithms::toposort`, this function traverses all
/// descendants of the root node.
pub(crate) fn toposort(g: &PortGraph, root: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
    let desc = pg::postorder(g, [root], Direction::Outgoing).collect::<BTreeSet<_>>();
    let port_filter = move |_, p| {
        if g.port_direction(p).expect("invalid port") == Direction::Outgoing {
            return true;
        }
        let Some(link) = g.port_link(p) else { return true };
        desc.contains(&g.port_node(link).expect("invalid port"))
    };
    pg::toposort_filtered(g, [root], Direction::Outgoing, |_| true, port_filter)
}
