use portgraph::proptest::gen_portgraph;
use portgraph::{Direction, PortGraph};
use proptest::prelude::*;

use super::depth::is_connected;

pub(crate) fn graph() -> PortGraph {
    let mut g = PortGraph::new();
    let v0 = g.add_node(2, 2);
    let v1 = g.add_node(2, 3);
    let vlol = g.add_node(3, 4);
    let v2 = g.add_node(2, 1);
    let v3 = g.add_node(2, 2);
    let v0_out0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    let v1_out1 = g.port_index(v1, 1, Direction::Outgoing).unwrap();
    let v2_in0 = g.port_index(v2, 0, Direction::Incoming).unwrap();
    let v2_in1 = g.port_index(v2, 1, Direction::Incoming).unwrap();
    let v2_out0 = g.port_index(v2, 0, Direction::Outgoing).unwrap();
    let v3_in1 = g.port_index(v3, 1, Direction::Incoming).unwrap();
    g.link_ports(v0_out0, v2_in1).unwrap();
    g.link_ports(v1_out1, v2_in0).unwrap();
    g.link_ports(v2_out0, v3_in1).unwrap();
    g.remove_node(vlol);
    g
}

pub(crate) fn gen_portgraph_connected(
    n_nodes: usize,
    n_ports: usize,
    max_edges: usize,
) -> impl Strategy<Value = PortGraph> {
    gen_portgraph(n_nodes, n_ports, max_edges).prop_filter("Must be connected", is_connected)
}
