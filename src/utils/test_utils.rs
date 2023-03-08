use std::fmt::Debug;
use std::{cmp, collections::BTreeSet, iter::zip};

use portgraph::{Direction, PortGraph, PortIndex};
use proptest::prelude::*;
use rand::seq::SliceRandom;

use super::depth::is_connected;

pub(crate) fn graph() -> PortGraph {
    let mut g = PortGraph::new();
    let v0 = g.add_node(2, 2);
    let v1 = g.add_node(2, 3);
    let vlol = g.add_node(3, 4);
    let v2 = g.add_node(2, 1);
    let v3 = g.add_node(2, 2);
    let v0_out0 = g.port(v0, 0, Direction::Outgoing).unwrap();
    let v1_out1 = g.port(v1, 1, Direction::Outgoing).unwrap();
    let v2_in0 = g.port(v2, 0, Direction::Incoming).unwrap();
    let v2_in1 = g.port(v2, 1, Direction::Incoming).unwrap();
    let v2_out0 = g.port(v2, 0, Direction::Outgoing).unwrap();
    let v3_in1 = g.port(v3, 1, Direction::Incoming).unwrap();
    g.link_ports(v0_out0, v2_in1).unwrap();
    g.link_ports(v1_out1, v2_in0).unwrap();
    g.link_ports(v2_out0, v3_in1).unwrap();
    g.remove_node(vlol);
    g
}

/// A strategy with `size` unique values from vals
fn unique_vec<T: Debug + Clone>(vals: Vec<T>, mut size: usize) -> impl Strategy<Value = Vec<T>> {
    size = cmp::min(size, vals.len());
    Just(vals)
        .prop_shuffle()
        .prop_map(move |v| v[..size].to_vec())
}

prop_compose! {
    fn no_edge_graph(max_n_nodes: usize, max_port: usize)(
        ports in prop::collection::vec(0..max_port, 2..=max_n_nodes)
    ) -> PortGraph {
        let mut g = PortGraph::new();
        let mut ind = 0;
        while ind + 1 < ports.len() {
            g.add_node(ports[ind], ports[ind + 1]);
            ind += 2;
        }
        g
    }
}

fn graph_and_edges(
    max_n_nodes: usize,
    max_port: usize,
    max_n_edges: usize,
) -> impl Strategy<Value = (PortGraph, Vec<PortIndex>, Vec<PortIndex>)> {
    let graph = no_edge_graph(max_n_nodes, max_port);
    (0..=max_n_edges, graph).prop_perturb(|(mut n_edges, graph), mut rng| {
        let mut in_ports = Vec::new();
        let mut out_ports = Vec::new();
        for p in graph.ports_iter() {
            match graph.port_direction(p).unwrap() {
                Direction::Incoming => in_ports.push(p),
                Direction::Outgoing => out_ports.push(p),
            }
        }
        in_ports.shuffle(&mut rng);
        out_ports.shuffle(&mut rng);

        n_edges = [n_edges, in_ports.len(), out_ports.len()]
            .into_iter()
            .min()
            .unwrap();
        if in_ports.len() > n_edges {
            in_ports.drain(n_edges..);
        }
        if out_ports.len() > n_edges {
            out_ports.drain(n_edges..);
        }
        (graph, in_ports, out_ports)
    })
}

prop_compose! {
    pub(crate) fn non_empty_portgraph(max_n_nodes: usize, max_port: usize, max_n_edges: usize)(
        (mut graph, in_stubs, out_stubs) in graph_and_edges(max_n_nodes, max_port, max_n_edges)
    ) -> PortGraph {
        for (incoming, outgoing) in zip(in_stubs, out_stubs) {
            graph.link_ports(outgoing, incoming).unwrap();
        }
        graph
    }
}

pub(crate) fn arb_portgraph_connected(
    n_nodes: usize,
    n_ports: usize,
    max_edges: usize,
) -> impl Strategy<Value = PortGraph> {
    non_empty_portgraph(n_nodes, n_ports, max_edges).prop_filter("Must be connected", is_connected)
}
