use std::collections::{BTreeMap, BTreeSet, VecDeque};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex};

/// Split `nodes` so that all its predecessors "go through" cover
pub fn cover_nodes<F, G>(
    graph: &mut PortGraph,
    cover: &BTreeSet<NodeIndex>,
    mut all_ports: BTreeSet<PortIndex>,
    mut clone_state: G,
    mut rekey: F,
) where
    F: FnMut(PortIndex, Option<PortIndex>, &PortGraph),
    G: FnMut(NodeIndex, NodeIndex, &PortGraph),
{
    let nodes: BTreeSet<_> = all_ports
        .iter()
        .map(|&p| graph.port_node(p).expect("Invalid port"))
        .collect();

    let mut curr_nodes: VecDeque<_> = cover
        .iter()
        .flat_map(|&n| {
            graph
                .output_links(n)
                .flatten()
                .map(|p| graph.port_node(p).expect("Invalid port"))
        })
        .filter(|n| nodes.contains(n))
        .collect();
    let mut visited = cover.clone();

    while let Some(node) = curr_nodes.pop_front() {
        if visited.contains(&node) {
            continue;
        } else if graph
            .input_links(node)
            .flatten()
            .map(|p| graph.port_node(p).expect("Invalid port"))
            .filter(|n| nodes.contains(n) && !visited.contains(n))
            .next()
            .is_some()
        {
            // there is a predecessor in nodes that was not yet visited, so wait
            curr_nodes.push_back(node);
            continue;
        } else {
            visited.insert(node);
        }
        let (within, without): (Vec<_>, Vec<_>) =
            graph.inputs(node).partition(|p| all_ports.contains(p));
        if !without.is_empty() && !within.is_empty() {
            split_node(
                graph,
                node,
                within,
                without,
                &mut clone_state,
                |old, new, graph| {
                    if all_ports.remove(&old) {
                        all_ports.insert(new.expect("Removed port"));
                    }
                    rekey(old, new, graph)
                },
            );
        }
        curr_nodes.extend(
            graph
                .output_links(node)
                .flatten()
                .map(|p| graph.port_node(p).expect("Invalid port"))
                .filter(|n| nodes.contains(n)),
        );
    }
}

/// Splits node1 into (node1, node2) according to partition
fn split_node<F, G>(
    graph: &mut PortGraph,
    node1: NodeIndex,
    mut in_ports1: Vec<portgraph::PortIndex>,
    in_ports2: Vec<portgraph::PortIndex>,
    mut clone_state: G,
    mut rekey: F,
) -> (NodeIndex, NodeIndex)
where
    F: FnMut(PortIndex, Option<PortIndex>, &PortGraph),
    G: FnMut(NodeIndex, NodeIndex, &PortGraph),
{
    let n_out = graph.num_outputs(node1);
    let node2 = graph.add_node(in_ports2.len(), n_out);

    // TODO maybe use compact_ports would be more concise, but would
    // require a refactor of portgraph to e.g. only compact input ports
    for (new_port_offset, in_port) in in_ports2.into_iter().enumerate() {
        let new_port = graph
            .port_index(node2, new_port_offset, Direction::Incoming)
            .expect("Just created");
        if let Some(out_port) = graph.unlink_port(in_port) {
            graph.link_ports(out_port, new_port).unwrap();
            rekey(in_port, new_port.into(), graph);
        }
    }
    // Compact input ports for node1
    in_ports1.sort_unstable();
    let num_in_ports1 = in_ports1.len();
    for (new_port_offset, in_port) in in_ports1.into_iter().enumerate() {
        let new_port = graph
            .port_index(node1, new_port_offset, Direction::Incoming)
            .expect("At most num_inputs iterations");
        if in_port != new_port {
            // invariant: new_port < port
            if let Some(out_port) = graph.unlink_port(in_port) {
                graph.link_ports(out_port, new_port).unwrap();
                rekey(in_port, new_port.into(), graph);
            }
        }
    }
    graph.set_num_ports(node1, num_in_ports1, graph.num_outputs(node1), &mut rekey);

    // Precompute the additional number of input ports needed
    let mut cnts = BTreeMap::<_, usize>::new();
    for next_port in graph.output_links(node1).flatten() {
        *cnts.entry(graph.port_node(next_port).unwrap()).or_default() += 1;
    }
    // Allocate the additional input ports once and for all
    let mut new_port_offset: BTreeMap<_, _> = cnts
        .keys()
        .map(|&node| (node, graph.num_inputs(node)))
        .collect();
    for (&node, &add_port) in cnts.iter() {
        graph.set_num_ports(
            node,
            graph.num_inputs(node) + add_port,
            graph.num_outputs(node),
            &mut rekey,
        );
    }

    // Copy each output link of node1 to node2
    for out_port in graph.outputs(node1) {
        if let Some(in_port) = graph.port_link(out_port) {
            let offset = graph.port_offset(out_port).unwrap();
            let out_port = graph
                .port_index(node2, offset, Direction::Outgoing)
                .expect("At most node1.num_outputs()");
            let in_node = graph.port_node(in_port).expect("Invalid port");
            let offset = new_port_offset.get_mut(&in_node).unwrap();
            let in_port = graph
                .port_index(in_node, *offset, Direction::Incoming)
                .expect("preallocated above");
            graph.link_ports(out_port, in_port).unwrap();
            *offset += 1;
        }
    }

    clone_state(node1, node2, graph);
    (node1, node2)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;
    use portgraph::{Direction, NodeIndex, PortGraph};

    use super::cover_nodes;

    #[test]
    fn test_cover() {
        let mut g = PortGraph::new();
        let n0_0 = g.add_node(0, 1);
        let n0_1 = g.add_node(0, 2);
        let n0_2 = g.add_node(0, 2);
        let n1_0 = g.add_node(2, 1);
        let n1_1 = g.add_node(2, 1);
        let n1_2 = g.add_node(1, 0);
        let n2 = g.add_node(2, 0);

        let edge = |g: &PortGraph, (n1, p1), (n2, p2)| {
            (
                g.port_index(n1, p1, Direction::Outgoing).unwrap(),
                g.port_index(n2, p2, Direction::Incoming).unwrap(),
            )
        };

        let mut left_edges = [
            edge(&g, (n0_0, 0), (n1_0, 0)),
            edge(&g, (n0_1, 0), (n1_0, 1)),
            edge(&g, (n0_1, 1), (n1_1, 0)),
            edge(&g, (n1_0, 0), (n2, 0)),
            edge(&g, (n1_1, 0), (n2, 1)),
        ];
        let right_edges = [
            edge(&g, (n0_2, 0), (n1_1, 1)),
            edge(&g, (n0_2, 1), (n1_2, 0)),
        ];
        for (out_p, in_p) in left_edges.iter().copied().chain(right_edges) {
            g.link_ports(out_p, in_p).unwrap();
        }

        let mut new_nodes = BTreeMap::new();
        let all_ports = [n0_0, n0_1, n1_0, n2]
            .into_iter()
            .flat_map(|n| g.inputs(n))
            .chain([g.input(n1_1, 0).unwrap()])
            .collect();
        cover_nodes(
            &mut g,
            &[n0_0, n0_1].into(),
            all_ports,
            |old_n, new_n, _| {
                new_nodes.insert(old_n, new_n);
            },
            |old_p, new_p, _| {
                for (p1, p2) in left_edges.iter_mut() {
                    if *p1 == old_p {
                        *p1 = new_p.unwrap();
                    }
                    if *p2 == old_p {
                        *p2 = new_p.unwrap();
                    }
                }
            },
        );

        assert_eq!(g.node_count(), 9);

        let new_n1_1 = new_nodes[&n1_1];
        assert_eq!(new_n1_1, NodeIndex::new(7));
        let new_n2 = new_nodes[&n2];
        assert_eq!(new_n2, NodeIndex::new(8));
        let new_right_edges = [
            edge(&g, (n0_2, 0), (new_n1_1, 0)),
            edge(&g, (n0_2, 1), (n1_2, 0)),
            edge(&g, (new_n1_1, 0), (new_n2, 0)),
        ];

        let edges = left_edges.into_iter().chain(new_right_edges).collect_vec();
        assert_eq!(edges.len(), g.link_count());
        for (out_p, in_p) in edges {
            assert_eq!(g.port_link(out_p), Some(in_p));
        }
    }
}
