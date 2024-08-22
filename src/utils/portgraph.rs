//! Graph-related utilities

use std::collections::{BTreeMap, VecDeque};

use crate::HashSet;

use petgraph::unionfind::UnionFind;
use portgraph::{Direction, LinkView, NodeIndex, PortGraph, PortIndex, PortOffset, PortView};

/// Partition the set of edge of a portgraph into lines.
///
/// A line is a non-empty sequence of edges such that the target port of the
/// n-th edge is opposite the source port of the n+1-th edge.
///
/// This assumes that the port graph is connected. Otherwise, only the connected
/// component of the `root` is traversed.
///
/// Note that it is important for this to work that the port labels are unique
/// for each edge. We therefore only support `PortGraph` and not the more
/// general `LinkView` trait.
pub(crate) fn line_partition(
    graph: &PortGraph,
    root: NodeIndex,
) -> Vec<Vec<(PortIndex, PortIndex)>> {
    let mut lines = Vec::new();
    let mut next_links = VecDeque::from_iter(graph.all_links(root));
    // The links already visited, where the pair of ports is viewed as a set.
    // Achieved by sorting the pair of ports in the order of their indices.
    let mut visited_links = HashSet::default();
    let mut visited_nodes = HashSet::from_iter([root]);
    while let Some(line_start) = next_links.pop_front() {
        if visited_links.contains(&as_ordered_pair(line_start)) {
            continue;
        }
        let mut curr_line = vec![line_start];
        visited_links.insert(as_ordered_pair(line_start));
        loop {
            let last_port = curr_line.last().unwrap().1;
            let curr_node = graph.port_node(last_port).unwrap();

            // Mark node as visited, add all outgoing links to the queue.
            if visited_nodes.insert(curr_node) {
                let new_links = graph
                    .all_links(curr_node)
                    .filter(|&l| !visited_links.contains(&as_ordered_pair(l)));
                next_links.extend(new_links);
            }

            // Get the next link to visit
            let next_port = {
                let offset = graph.port_offset(last_port).unwrap();
                let opp_offset = match offset.direction() {
                    Direction::Incoming => PortOffset::new_outgoing(offset.index()),
                    Direction::Outgoing => PortOffset::new_incoming(offset.index()),
                };
                graph.port_index(curr_node, opp_offset)
            };
            let Some(left_port) = next_port else {
                break;
            };
            let Some(right_port) = graph.port_link(left_port) else {
                break;
            };

            // Add link to current line and mark as visited
            if visited_links.insert(as_ordered_pair((left_port, right_port))) {
                curr_line.push((left_port, right_port));
            } else {
                break;
            }
        }
        lines.push(curr_line);
    }
    lines
}

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

fn as_ordered_pair(link: (PortIndex, PortIndex)) -> (PortIndex, PortIndex) {
    if link.0 < link.1 {
        return link;
    }
    (link.1, link.0)
}

#[cfg(test)]
mod tests {
    use portgraph::{LinkMut, PortMut};

    use super::*;

    #[test]
    fn test_line_partition() {
        let mut g = PortGraph::new();
        let n = g.add_node(0, 1);
        let m = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(m, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let lines = line_partition(&g, n);
        assert_eq!(
            lines,
            vec![vec![(
                g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
                g.port_index(m, PortOffset::new_incoming(0)).unwrap(),
            ),]]
        );
    }
}
