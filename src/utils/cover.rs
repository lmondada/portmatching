use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

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
    G: FnMut(NodeIndex, NodeIndex, &PortGraph),
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
        let mut next_nodes = graph
            .output_links(node)
            .flatten()
            .map(|p| graph.port_node(p).expect("Invalid port"))
            .filter(|&n| all_nodes.contains(&n))
            .collect();
        if ins.len() > 1 {
            // Drain the ports that we are tracking in all_ports and new_in_ports
            // so we can keep track of them during the node splitting
            let all_ports_out: RefCell<BTreeSet<_>> = RefCell::new(
                graph
                    .outputs(node)
                    .filter(|&p| {
                        let in_port = graph.port_link(p).expect("Disconnected port");
                        all_ports.remove(&in_port).is_some()
                    })
                    .collect(),
            );
            let new_in_ports_out: RefCell<BTreeSet<_>> = RefCell::new(
                graph
                    .outputs(node)
                    .filter(|&p| {
                        let in_port = graph.port_link(p).expect("Disconnected port");
                        new_in_ports.remove(&in_port)
                    })
                    .collect(),
            );
            let clone_state = |old, new, graph: &PortGraph| {
                if visited.contains(&old) {
                    visited.insert(new);
                }
                if all_nodes.contains(&old) {
                    all_nodes.insert(new);
                }
                for old_out in graph.outputs(old) {
                    let offset = graph.port_offset(old_out).expect("invalid port");
                    let new_out = graph.port_index(new, offset).expect("invalid offset");
                    let mut all_ports_out = all_ports_out.borrow_mut();
                    if all_ports_out.contains(&old_out) {
                        all_ports_out.insert(new_out);
                    }
                    let mut new_in_ports_out = new_in_ports_out.borrow_mut();
                    if new_in_ports_out.contains(&old_out) {
                        new_in_ports_out.insert(new_out);
                    }
                }
                clone_state(old, new, graph)
            };
            // Split node according to ins/outs partition
            let new_outs = split_node(graph, node, &ins, &outs, clone_state, |old, new| {
                let mut all_ports_out = all_ports_out.borrow_mut();
                if all_ports_out.remove(&old) {
                    new.map(|new| all_ports_out.insert(new));
                }
                let mut new_in_ports_out = new_in_ports_out.borrow_mut();
                if new_in_ports_out.remove(&old) {
                    new.map(|new| new_in_ports_out.insert(new));
                }
                rekey(old, new)
            });
            let all_ports_out = all_ports_out.into_inner();
            let new_in_ports_out = new_in_ports_out.into_inner();
            // Update `all_ports` with the new ports
            for (out_ports, l) in new_outs.iter().zip(keys) {
                let Some(l) = l else { continue };
                for &out_port in out_ports {
                    if all_ports_out.contains(&out_port) {
                        let in_port = graph.port_link(out_port).expect("Disconnected port");
                        if node == NodeIndex::new(1) {
                            println!("inserting {:?} -> {}", in_port, l);
                        }
                        all_ports.insert(in_port, vec![l]);
                    }
                }
            }
            // Update `new_in_ports` with the new ports
            new_in_ports.extend(
                new_in_ports_out
                    .into_iter()
                    .map(|p| graph.port_link(p).expect("Disconnected port")),
            );
        }
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

/// Splits `old_node` into multiple nodes according to ins/outs partition
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
    G: FnMut(NodeIndex, NodeIndex, &PortGraph),
{
    let n_partitions = ins.len();
    let num_in = graph.num_inputs(old_node);
    let num_out = graph.num_outputs(old_node);
    assert_eq!(n_partitions, outs.len());

    let mut nodes = Vec::with_capacity(n_partitions);
    nodes.push(old_node);

    // Clone `old_node` n_partitions - 1 times
    for i in 1..n_partitions {
        nodes.push(graph.add_node(num_in, num_out));
        clone_state(old_node, nodes[i], graph);
    }

    // Rewire all input ports (do `old_node` last, so that the ports are freed)
    for (&node, in_ports) in nodes.iter().zip(ins).rev() {
        let out_ports = in_ports
            .iter()
            .map(|&p| graph.unlink_port(p).expect("Disconnected port"))
            .collect::<Vec<_>>()
            .into_iter();
        let new_in_ports = (0..in_ports.len())
            .map(|i| {
                graph
                    .port_index(node, PortOffset::new_incoming(i))
                    .expect("in_ports.len() <= num_in")
            })
            .collect::<Vec<_>>();
        for ((new_out, new_in), &old_in) in out_ports.zip(new_in_ports).zip(in_ports) {
            graph.link_ports(new_out, new_in).unwrap();
            if old_in != new_in {
                rekey(old_in, new_in.into());
            }
        }
    }

    // Precompute the number of additional input ports needed (on the children nodes)
    // Unlike the inputs, the output ports can be repeated in the "partition",
    // so that creating new ports in the children nodes may be necessary
    let mut cnts = BTreeMap::<_, usize>::new();
    let mut outport_seen = BTreeSet::new();
    for &out_port in outs.iter().flatten() {
        if outport_seen.insert(out_port) {
            continue;
        }
        let next_port = graph.port_link(out_port).expect("Disconnected port");
        *cnts.entry(graph.port_node(next_port).unwrap()).or_default() += 1;
    }
    // Allocate the additional input ports on the children nodes
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

    // Rewire all output links
    // Store in `links` the original output port -> input port mapping
    let links: BTreeMap<_, _> = graph
        .outputs(old_node)
        .map(|p| (p, graph.unlink_port(p).expect("Disconnected port")))
        .collect();
    for (&node, out_ports) in nodes.iter().zip(outs) {
        let in_ports = out_ports
            .iter()
            .map(|&p| {
                let in_port = links[&p];
                if graph.port_link(in_port).is_none() {
                    // in_port is free, use it
                    in_port
                } else {
                    // create new in_port
                    let in_node = graph.port_node(in_port).expect("Invalid port");
                    let offset = new_port_offset.get_mut(&in_node).unwrap();
                    let new_in_port = graph
                        .port_index(in_node, PortOffset::new_incoming(*offset))
                        .expect("preallocated above");
                    *offset += 1;
                    new_in_port
                }
            })
            .collect::<Vec<_>>()
            .into_iter();
        let new_out_ports = out_ports
            .iter()
            .map(|&p| {
                let offset = graph.port_offset(p).expect("Invalid port");
                graph.port_index(node, offset).expect("same sig")
            })
            .collect::<Vec<_>>()
            .into_iter();
        for (new_out, new_in) in new_out_ports.zip(in_ports) {
            graph.link_ports(new_out, new_in).unwrap();
        }
    }

    // Compactify output ports by moving all linked ports to beginning
    for &node in nodes.iter() {
        let linked_ports = graph
            .outputs(node)
            .filter(|&p| graph.port_link(p).is_some())
            .collect::<Vec<_>>() // store iterator so that `graph` can be mut
            .into_iter();
        for old_out in linked_ports {
            let Some(new_out) = graph
                .outputs(node)
                .find(|&p| graph.port_link(p).is_none())
            else { break };
            let in_port = graph.unlink_port(old_out).expect("is linked");
            graph.link_ports(new_out, in_port).expect("is free");
            rekey(old_out, new_out.into());
        }
    }

    // Truncate surplus ports
    for ((&node, ins), outs) in nodes.iter().zip(ins).zip(outs) {
        graph.set_num_ports(node, ins.len(), outs.len(), &mut rekey);
    }

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
            |old_n, new_n, _| {
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
