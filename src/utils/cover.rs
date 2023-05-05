use std::collections::{BTreeMap, BTreeSet, VecDeque};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::graph_tries::root_state;

/// Extract new threads into separate nodes.
///
/// TODO
pub fn untangle_threads<F, G>(
    graph: &mut PortGraph,
    all_threads: BTreeSet<(PortIndex, usize)>,
    mut new_in_ports: BTreeSet<PortIndex>,
    start_states: &BTreeMap<NodeIndex, Vec<usize>>,
    mut clone_state: G,
    mut rekey: F,
) -> BTreeSet<NodeIndex>
where
    F: FnMut(PortIndex, Option<PortIndex>),
    G: FnMut(NodeIndex, NodeIndex, &Vec<PortIndex>, &PortGraph),
{
    // Map from all in-ports along one thread to its layers
    let mut all_ports = BTreeMap::new();
    for (k, v) in all_threads {
        all_ports.entry(k).or_insert_with(Vec::new).push(v);
    }
    let mut all_nodes = all_ports
        .keys()
        .map(|&p| graph.port_node(p).expect("Invalid port"))
        .collect::<BTreeSet<_>>();
    if all_nodes.is_empty() {
        return [root_state()].into();
    }

    let mut curr_nodes: VecDeque<_> = graph
        .output_links(root_state())
        .flatten()
        .map(|p| graph.port_node(p).expect("Invalid port"))
        .collect();
    let mut visited: BTreeSet<_> = [root_state()].into();

    while let Some(node) = curr_nodes.pop_front() {
        if visited.contains(&node) {
            continue;
        } else if graph
            .input_links(node)
            .flatten()
            .map(|p| graph.port_node(p).expect("Invalid port"))
            .any(|n| all_nodes.contains(&n) && !visited.contains(&n))
        {
            // there is a predecessor in nodes that was not yet visited, so wait
            curr_nodes.push_back(node);
            continue;
        } else {
            visited.insert(node);
        }
        let mut ins = BTreeMap::new();
        for p in graph.inputs(node) {
            let k = all_ports.get(&p).map(|v| {
                assert_eq!(v.len(), 1);
                v[0]
            });
            ins.entry(k).or_insert_with(Vec::new).push(p);
        }
        let keys: Vec<_> = ins.keys().copied().collect();
        let ins: Vec<_> = ins.into_values().collect();
        let outs: Vec<_> = keys
            .iter()
            .map(|&(mut l)| {
                // Increase tolerance if we are in a start state
                if let Some(v) = start_states.get(&node) {
                    if let Some(l) = l.as_mut() {
                        if v.contains(l) {
                            *l += 1;
                        }
                    }
                }
                graph
                    .outputs(node)
                    .filter(|&p| {
                        let in_port = graph.port_link(p).expect("Disconnected port");
                        if !new_in_ports.contains(&in_port) {
                            return true;
                        }
                        let Some(v) = all_ports.get(&in_port) else { return true };
                        let Some(l) = l else { return false };
                        v.contains(&l)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut next_nodes = if ins.len() > 1 {
            // backup which ports in `outs` are in all_ports
            let outs_in_all_ports = outs
                .iter()
                .map(|ps| {
                    ps.iter()
                        .map(|&p| {
                            let in_port = graph.port_link(p).expect("Disconnected port");
                            all_ports.contains_key(&in_port)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            // Remove old ports from `all_ports`
            for old_p in outs.iter().flatten() {
                let old_in_port = graph.port_link(*old_p).expect("Disconnected port");
                all_ports.remove(&old_in_port);
            }
            // Split node according to ins/outs partition
            let new_outs = split_node(graph, node, &ins, &outs, &mut clone_state, |old, new| {
                if let Some(val) = all_ports.remove(&old) {
                    all_ports.insert(new.expect("Removed port"), val);
                }
                if new_in_ports.remove(&old) {
                    new_in_ports.insert(new.expect("Removed port"));
                }
                rekey(old, new)
            });
            // Update `all_ports` with the new ports
            all_ports.extend(
                new_outs
                    .iter()
                    .zip(keys.iter())
                    .zip(outs_in_all_ports)
                    .flat_map(|((new_ps, l), pred)| new_ps.iter().map(move |p| (p, l)).zip(pred))
                    .filter_map(|(fst, pred)| pred.then_some(fst))
                    .filter_map(|(&new, &l)| {
                        let in_port = graph.port_link(new).expect("Disconnected port");
                        l.map(|l| (in_port, vec![l]))
                    }),
            );
            // splitting nodes creates new nodes, so we must update `visited`
            // and `all_nodes`
            let new_outs = if keys.first() == Some(&None) {
                &new_outs[1..]
            } else {
                &new_outs[..]
            };
            let new_nodes = new_outs
                .iter()
                .flatten()
                .map(|&p| graph.port_node(p).expect("Invalid port"));
            visited.extend(new_nodes.clone());
            all_nodes.remove(&node);
            all_nodes.extend(new_nodes);
            // compute the next nodes
            new_outs
                .iter()
                .flatten()
                .map(|&p| {
                    let in_port = graph.port_link(p).expect("Disconnected port");
                    graph.port_node(in_port).expect("Invalid port")
                })
                .filter(|&n| all_nodes.contains(&n))
                .collect()
        } else {
            graph
                .output_links(node)
                .flatten()
                .map(|p| graph.port_node(p).expect("Invalid port"))
                .filter(|&n| all_nodes.contains(&n))
                .collect()
        };
        curr_nodes.append(&mut next_nodes);
    }
    all_nodes
        .iter()
        .copied()
        .filter(|&n| {
            graph
                .output_links(n)
                .flatten()
                .all(|p| !all_ports.contains_key(&p))
        })
        .collect()
}

/// Splits node1 into multiple nodes according to ins/outs partition
fn split_node<F, G>(
    graph: &mut PortGraph,
    old_node: NodeIndex,
    ins: &Vec<Vec<PortIndex>>,
    outs: &Vec<Vec<PortIndex>>,
    mut clone_state: G,
    mut rekey: F,
) -> Vec<Vec<PortIndex>>
where
    F: FnMut(PortIndex, Option<PortIndex>),
    G: FnMut(NodeIndex, NodeIndex, &Vec<PortIndex>, &PortGraph),
{
    let mut nodes = Vec::with_capacity(ins.len());
    for (in_ports, out_ports) in ins.iter().zip(outs) {
        nodes.push(graph.add_node(in_ports.len(), out_ports.len()));
    }

    // Rewire all input ports
    for (&node, in_ports) in nodes.iter().zip(ins) {
        for (i, &in_port) in in_ports.iter().enumerate() {
            let new_port = graph
                .port_index(node, PortOffset::new_incoming(i))
                .expect("Just created");
            if let Some(out_port) = graph.unlink_port(in_port) {
                graph.link_ports(out_port, new_port).unwrap();
                rekey(in_port, new_port.into());
            }
        }
    }

    // Precompute the number of additional input ports needed (on the children nodes)
    let mut cnts = BTreeMap::<_, usize>::new();
    let mut outport_seen = BTreeSet::new();
    for &out_port in outs.iter().flatten() {
        if outport_seen.insert(out_port) {
            continue;
        }
        let next_port = graph.port_link(out_port).expect("Disconnected port");
        *cnts.entry(graph.port_node(next_port).unwrap()).or_default() += 1;
    }
    // Allocate the additional input ports once
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

    // Insert all new output links
    let mut links = BTreeMap::new();
    for (&node, out_ports) in nodes.iter().zip(outs) {
        for (i, &out_port) in out_ports.iter().enumerate() {
            let new_out_port = graph.output(node, i).expect("precomputed num_outputs");
            if let Some(in_port) = graph.unlink_port(out_port) {
                graph.link_ports(new_out_port, in_port).unwrap();
                links.insert(out_port, in_port);
            } else {
                let in_port = links[&out_port];
                let in_node = graph.port_node(in_port).expect("Invalid port");
                let offset = new_port_offset.get_mut(&in_node).unwrap();
                let new_in_port = graph
                    .port_index(in_node, PortOffset::new_incoming(*offset))
                    .expect("preallocated above");
                *offset += 1;
                graph.link_ports(new_out_port, new_in_port).unwrap();
            }
        }
        clone_state(old_node, node, out_ports, graph);
    }

    graph.remove_node(old_node);

    nodes
        .into_iter()
        .map(|n| graph.outputs(n).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use portgraph::{PortGraph, PortOffset};

    use super::untangle_threads;

    #[test]
    fn test_cover() {
        let mut g = PortGraph::new();
        let nodes = vec![
            g.add_node(0, 2),
            g.add_node(2, 1),
            g.add_node(1, 2),
            g.add_node(2, 1),
            g.add_node(1, 1),
            g.add_node(1, 0),
        ];

        let edge = |g: &PortGraph, (n1, p1), (n2, p2)| {
            (
                g.port_index(nodes[n1], PortOffset::new_outgoing(p1))
                    .unwrap(),
                g.port_index(nodes[n2], PortOffset::new_incoming(p2))
                    .unwrap(),
            )
        };

        let mut threads = [
            edge(&g, (0, 0), (1, 0)),
            edge(&g, (0, 1), (2, 0)),
            edge(&g, (1, 0), (3, 0)),
            edge(&g, (2, 0), (1, 1)),
            edge(&g, (3, 0), (5, 0)),
        ];
        let thread_inds = [0, 0, 1, 1, 2];
        let other_edges = [edge(&g, (2, 1), (4, 0)), edge(&g, (4, 0), (3, 1))];
        for (out_p, in_p) in threads.iter().copied().chain(other_edges) {
            g.link_ports(out_p, in_p).unwrap();
        }

        let mut new_nodes = BTreeMap::new();
        let all_ports = threads
            .iter()
            .zip(thread_inds)
            .map(|(&(_, p), i)| (p, i))
            .collect();
        untangle_threads(
            &mut g,
            all_ports,
            [].into(),
            &[
                (nodes[1], [0, 1].into()),
                (nodes[2], [0].into()),
                (nodes[3], [1, 2].into()),
            ]
            .into(),
            |old_n, new_n, _, _| {
                new_nodes.insert(old_n, new_n);
            },
            |old_p, new_p| {
                for (p1, p2) in threads.iter_mut() {
                    if *p1 == old_p {
                        *p1 = new_p.unwrap();
                    }
                    if *p2 == old_p {
                        *p2 = new_p.unwrap();
                    }
                }
            },
        );

        assert_eq!(g.node_count(), 11);
    }
}
