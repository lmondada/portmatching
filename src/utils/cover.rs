use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    mem,
};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset, SecondaryMap};

use crate::graph_tries::root_state;

use super::rekey_secmap;

/// Extract new threads into separate nodes.
///
/// TODO
pub fn untangle_threads<F, G>(
    graph: &mut PortGraph,
    mut trace: SecondaryMap<PortIndex, (Vec<usize>, bool)>,
    mut clone_state: G,
    mut rekey: F,
) -> BTreeSet<NodeIndex>
where
    F: FnMut(PortIndex, Option<PortIndex>),
    G: FnMut(NodeIndex, NodeIndex, &PortGraph),
{
    // All nodes that are traversed by at least one thread
    let mut all_nodes = graph
        .nodes_iter()
        .filter(|&n| {
            graph.all_ports(n).any(|p| {
                let (vec, _) = &trace[p];
                !vec.is_empty()
            })
        })
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
        // Organise inports by their layer ind
        let mut ins = BTreeMap::new();
        for p in graph.inputs(node) {
            let vec = trace[p].0.as_slice();
            assert!(vec.len() <= 1);
            let k = vec.first().copied();
            ins.entry(k).or_insert_with(Vec::new).push(p);
        }
        let keys: Vec<_> = ins.keys().copied().collect();
        let ins: Vec<_> = ins.into_values().collect();
        let outs: Vec<_> = keys
            .iter()
            .map(|&l| {
                graph
                    .outputs(node)
                    .filter(|&p| {
                        let vec = trace[p].0.as_slice();
                        let is_new = trace[p].1;
                        if !is_new {
                            return true;
                        }
                        let Some(l) = l else { return false };
                        vec.contains(&l)
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
            all_nodes.remove(&node);
            let mut trace_next_in = SecondaryMap::new();
            for out_port in graph.outputs(node) {
                let in_port = graph.port_link(out_port).expect("Disconnected port");
                trace_next_in[out_port] = mem::take(&mut trace[in_port]);
            }
            let trace_mut = RefCell::new(&mut trace);
            let trace_curr_mut = RefCell::new(&mut trace_next_in);
            let clone_state = |old, new, graph: &PortGraph| {
                let mut trace = trace_mut.borrow_mut();
                let mut trace_curr = trace_curr_mut.borrow_mut();
                if visited.contains(&old) {
                    visited.insert(new);
                }
                for old_port in graph.all_ports(old) {
                    let offset = graph.port_offset(old_port).expect("invalid port");
                    let new_port = graph.port_index(new, offset).expect("invalid offset");
                    trace[new_port] = trace[old_port].clone();
                    trace_curr[new_port] = trace_curr[old_port].clone();
                }
                clone_state(old, new, graph)
            };
            // Split node according to ins/outs partition
            let new_nodes = split_node(graph, node, &ins, &outs, clone_state, |old, new| {
                let mut trace = trace_mut.borrow_mut();
                let mut trace_curr = trace_curr_mut.borrow_mut();
                rekey_secmap(&mut trace, old, new);
                rekey_secmap(&mut trace_curr, old, new);
                rekey(old, new)
            });
            // Restore trace for next inputs
            for out_port in new_nodes.iter().flat_map(|&n| graph.outputs(n)) {
                let in_port = graph.port_link(out_port).expect("Disconnected port");
                trace[in_port] = mem::take(&mut trace_next_in[out_port]);
            }
            // Reduce the inds of the new nodes
            for (k, n) in keys.iter().zip(new_nodes) {
                if k.is_some() {
                    all_nodes.insert(n);
                }
                for out_port in graph.outputs(n) {
                    let in_port = graph.port_link(out_port).expect("Disconnected port");
                    let [(out_trace, _), (in_trace, _)] = trace
                        .get_disjoint_mut([out_port, in_port])
                        .expect("linked ports must be disjoint");
                    let pos = k.and_then(|k| out_trace.iter().position(|&x| x == k));
                    *out_trace = pos.map(|pos| vec![out_trace[pos]]).unwrap_or_default();
                    *in_trace = pos.map(|pos| vec![in_trace[pos]]).unwrap_or_default();
                }
            }
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
                .all(|p| trace[p].0.is_empty())
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
) -> Vec<NodeIndex>
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
            if new_out < old_out {
                let in_port = graph.unlink_port(old_out).expect("is linked");
                graph.link_ports(new_out, in_port).expect("is free");
                rekey(old_out, new_out.into());
            }
        }
    }

    // Truncate surplus ports
    for ((&node, ins), outs) in nodes.iter().zip(ins).zip(outs) {
        graph.set_num_ports(node, ins.len(), outs.len(), &mut rekey);
    }

    debug_assert!(graph
        .ports_iter()
        .all(|port| graph.port_link(port).is_some()));

    nodes
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use portgraph::{PortGraph, PortOffset, SecondaryMap};

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

        let threads = [
            edge(&g, (0, 0), (1, 0)),
            edge(&g, (0, 1), (2, 0)),
            edge(&g, (1, 0), (3, 0)),
            edge(&g, (2, 0), (1, 1)),
            edge(&g, (3, 0), (5, 0)),
        ];
        let other_edges = [edge(&g, (2, 1), (4, 0)), edge(&g, (4, 0), (3, 1))];
        for (out_p, in_p) in threads.iter().copied().chain(other_edges) {
            g.link_ports(out_p, in_p).unwrap();
        }

        let mut new_nodes = BTreeMap::new();
        let mut trace: SecondaryMap<_, _> = Default::default();
        let thread_inds = [
            (vec![0], vec![1]),
            (vec![0], vec![1]),
            (vec![1, 2], vec![2, 3]),
            (vec![1], vec![2]),
            (vec![2, 3], vec![3, 4]),
        ];
        for (&(out_port, in_port), (out_ind, in_ind)) in threads.iter().zip(thread_inds) {
            trace[in_port] = (in_ind, false);
            trace[out_port] = (out_ind, false);
        }
        untangle_threads(
            &mut g,
            trace,
            |old_n, new_n, _| {
                new_nodes.insert(old_n, new_n);
            },
            |_, _| {},
        );

        assert_eq!(g.node_count(), 11);
    }
}
