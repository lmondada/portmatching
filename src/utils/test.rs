//! Utilities for testing.

use std::borrow::Borrow;
use std::fmt;

use itertools::Itertools;
use portgraph::{
    LinkMut, LinkView, MultiPortGraph, NodeIndex, PortGraph, PortMut, PortOffset, PortView,
};
use serde::{Deserialize, Serialize};

use crate::portgraph::indexing::PGIndexKey;
use crate::{IndexMap, PatternID, PatternMatch};

use super::portgraph::connected_components;

/// A minimalist version of a port graph pattern match, for testing purposes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SerialPatternMatch {
    pattern: PatternID,
    root: NodeIndex,
}

impl<S: IndexMap<Key = PGIndexKey, Value = NodeIndex>> From<PatternMatch<S>>
    for SerialPatternMatch
{
    fn from(value: PatternMatch<S>) -> Self {
        let pattern = value.pattern;
        let &root = value.match_data.get(&PGIndexKey::root(0)).unwrap().borrow();
        Self { pattern, root }
    }
}

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

#[cfg(feature = "proptest")]
pub use self::proptests::*;

#[cfg(feature = "proptest")]
mod proptests {
    use super::*;
    use portgraph::proptest::{gen_multiportgraph, gen_portgraph};
    use proptest::prelude::*;

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

    /// Proptest strategy for generating a connected portgraph.
    pub fn gen_portgraph_connected(
        n_nodes: usize,
        n_ports: usize,
        max_edges: usize,
    ) -> impl Strategy<Value = PortGraph> {
        connected_strat(gen_portgraph(n_nodes, n_ports, max_edges))
    }

    /// Proptest strategy for generating a connected multiportgraph.
    pub fn gen_multiportgraph_connected(
        n_nodes: usize,
        n_ports: usize,
        max_edges: usize,
    ) -> impl Strategy<Value = MultiPortGraph> {
        connected_strat(gen_multiportgraph(n_nodes, n_ports, max_edges))
    }
}
