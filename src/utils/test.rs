use std::fmt;

use itertools::Itertools;
use portgraph::proptest::{gen_multiportgraph, gen_portgraph};
use portgraph::{LinkMut, LinkView, MultiPortGraph, PortGraph, PortMut, PortOffset, PortView};
use proptest::prelude::*;

use super::connected_components;

pub(crate) fn graph() -> PortGraph {
    let mut g = PortGraph::new();
    let v0 = g.add_node(2, 2);
    let v1 = g.add_node(2, 3);
    let vlol = g.add_node(3, 4);
    let v2 = g.add_node(2, 1);
    let v3 = g.add_node(2, 2);
    let v0_out0 = g.port_index(v0, PortOffset::new_outgoing(0)).unwrap();
    let v1_out1 = g.port_index(v1, PortOffset::new_outgoing(1)).unwrap();
    let v2_in0 = g.port_index(v2, PortOffset::new_incoming(0)).unwrap();
    let v2_in1 = g.port_index(v2, PortOffset::new_incoming(1)).unwrap();
    let v2_out0 = g.port_index(v2, PortOffset::new_outgoing(0)).unwrap();
    let v3_in1 = g.port_index(v3, PortOffset::new_incoming(1)).unwrap();
    g.link_ports(v0_out0, v2_in1).unwrap();
    g.link_ports(v1_out1, v2_in0).unwrap();
    g.link_ports(v2_out0, v3_in1).unwrap();
    g.remove_node(vlol);
    g
}

/// Strategy adaptor to return the largest connected component of a graph.
fn connected_strat<G: PortView + LinkView + PortMut + fmt::Debug>(
    strat: impl Strategy<Value = G>,
) -> impl Strategy<Value = G> {
    strat.prop_map(|mut g| {
        let cc = connected_components(&g);
        let Some(max_cc) = cc.iter().position_max_by_key(|c| c.len()) else {
            return g;
        };
        for (i, c) in cc.into_iter().enumerate() {
            if i != max_cc {
                for v in c {
                    g.remove_node(v);
                }
            }
        }
        g
    })
}

pub(crate) fn gen_portgraph_connected(
    n_nodes: usize,
    n_ports: usize,
    max_edges: usize,
) -> impl Strategy<Value = PortGraph> {
    connected_strat(gen_portgraph(n_nodes, n_ports, max_edges))
}

pub(crate) fn gen_multiportgraph_connected(
    n_nodes: usize,
    n_ports: usize,
    max_edges: usize,
) -> impl Strategy<Value = MultiPortGraph> {
    connected_strat(gen_multiportgraph(n_nodes, n_ports, max_edges))
}
