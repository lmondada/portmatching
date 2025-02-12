//! Utilities for testing.

use std::borrow::Borrow;
use std::fmt;

use itertools::Itertools;
use portgraph::{LinkView, MultiPortGraph, NodeIndex, PortGraph, PortMut, PortView};
use serde::{Deserialize, Serialize};

use crate::concrete::portgraph::indexing::PGIndexKey;
use crate::indexing::Binding;
use crate::{BindMap, PatternID, PatternMatch};

use super::portgraph::connected_components;

/// A minimalist version of a port graph pattern match, for testing purposes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SerialPatternMatch {
    pattern: PatternID,
    root: NodeIndex,
}

impl<S: BindMap<Key = PGIndexKey, Value = NodeIndex>> From<PatternMatch<S>> for SerialPatternMatch {
    fn from(value: PatternMatch<S>) -> Self {
        let pattern = value.pattern;
        let Binding::Bound(root) = value.match_data.get_binding(&PGIndexKey::root(0)) else {
            panic!("unboud root value");
        };
        Self {
            pattern,
            root: *root.borrow(),
        }
    }
}

pub use self::proptests::*;

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
