use std::collections::BTreeMap;

use bimap::BiBTreeMap;
use portgraph::{NodeIndex, PortGraph};

use crate::pattern::{Edge, Pattern};

use super::Matcher;

pub struct SinglePatternMatcher {
    pattern: Pattern,
    edges: Vec<Edge>,
}

impl<'graph> Matcher<'graph> for SinglePatternMatcher {
    type Match = BTreeMap<NodeIndex, NodeIndex>;

    fn find_anchored_matches(&self, graph: &'graph PortGraph, anchor: NodeIndex) -> Vec<Self::Match> {
        self.find_anchored_match(graph, anchor)
            .map(|m| vec![m])
            .unwrap_or_default()
    }
}

impl SinglePatternMatcher {
    pub fn from_pattern(pattern: Pattern) -> Self {
        // This is our "matching recipe" -- we precompute it once and store it
        let edges = pattern.canonical_edge_ordering();
        Self { pattern, edges }
    }

    /// For single patterns, the anchored match, if it exists, is unique
    fn find_anchored_match(
        &self,
        host: &PortGraph,
        anchor: NodeIndex,
    ) -> Option<BTreeMap<NodeIndex, NodeIndex>> {
        let mut match_map = BiBTreeMap::from_iter([(self.root(), anchor)]);
        for &Edge(out_port, in_port) in self.edges.iter() {
            // Follow outgoing port...
            let out_node = self.graph_ref().port_node(out_port).unwrap();
            let &out_node_host = match_map.get_by_left(&out_node).unwrap();
            let out_offset = self.graph_ref().port_offset(out_port).unwrap();
            let out_port_host = host.port_index(out_node_host, out_offset)?;

            // ...into a new incoming port
            let Some(in_port) = in_port else {
                // Nothing to do
                continue;
            };
            let in_node = self.graph_ref().port_node(in_port).unwrap();
            let in_port_host = host.port_link(out_port_host)?;
            let in_node_host = host.port_node(in_port_host).unwrap();

            // Check that the in-port index is correct
            let in_offset = self.graph_ref().port_offset(in_port).unwrap();
            let in_offset_host = host.port_offset(in_port_host).unwrap();
            if in_offset != in_offset_host {
                return None;
            }
            if match_map.get_by_left(&in_node) != Some(&in_node_host) {
                // This will fail if the map is not injective
                match_map.insert_no_overwrite(in_node, in_node_host).ok()?
            }
        }
        BTreeMap::from_iter(match_map).into()
    }

    fn root(&self) -> NodeIndex {
        self.pattern.root
    }

    fn graph_ref(&self) -> &PortGraph {
        &self.pattern.graph
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use itertools::Itertools;
    use portgraph::{NodeIndex, PortGraph, PortOffset};

    use crate::{matcher::Matcher, pattern::Pattern, utils::test_utils::graph};

    use super::SinglePatternMatcher;

    #[test]
    fn single_pattern_match_simple() {
        let g = graph();
        let p = Pattern::from_graph(g.clone()).unwrap();
        let matcher = SinglePatternMatcher::from_pattern(p);

        let (n0, n1, n3, n4) = (
            NodeIndex::new(0),
            NodeIndex::new(1),
            NodeIndex::new(3),
            NodeIndex::new(4),
        );
        assert_eq!(
            matcher.find_matches(&g),
            vec![BTreeMap::from([(n0, n0), (n1, n1), (n3, n3), (n4, n4)])]
        );
    }

    #[test]
    fn single_pattern_distinguish_input_output() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_graph(g.clone()).unwrap();
        let matcher = SinglePatternMatcher::from_pattern(p);
        let mut g = PortGraph::new();
        g.add_node(1, 0);

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    #[test]
    fn single_node_loop() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_graph(g).unwrap();
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        let n = g.add_node(2, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g).len(), 1);
    }

    #[test]
    fn single_node_loop_no_match() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_graph(g).unwrap();
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n1, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    fn add_pattern(graph: &mut PortGraph, vertices: &[NodeIndex; 4]) {
        let [_, _, v2_in, v3_in] = vertices.map(|n| graph.inputs(n).collect_vec());
        let [v0_out, v1_out, v2_out, _] = vertices.map(|n| graph.outputs(n).collect_vec());

        graph.link_ports(v0_out[0], v2_in[1]).unwrap();
        graph.link_ports(v1_out[1], v2_in[0]).unwrap();
        graph.link_ports(v2_out[0], v3_in[1]).unwrap();
        graph.link_ports(v1_out[2], v3_in[2]).unwrap();
    }

    #[test]
    fn single_pattern_match_complex() {
        let mut pattern = PortGraph::new();
        for _ in 0..4 {
            pattern.add_node(3, 3);
        }
        let pi = |i| pattern.nodes_iter().nth(i).unwrap();
        let ps = [pi(0), pi(1), pi(2), pi(3)];
        add_pattern(&mut pattern, &ps);
        let p = Pattern::from_graph(pattern).unwrap();
        let matcher = SinglePatternMatcher::from_pattern(p);

        let mut g = PortGraph::new();
        for _ in 0..100 {
            g.add_node(3, 3);
        }
        let vi = |i| g.nodes_iter().nth(i).unwrap();
        let vs1 = [vi(0), vi(10), vi(30), vi(55)];
        let vs2 = [vi(23), vi(3), vi(44), vi(12)];
        let vs3 = [vi(55), vi(12), vi(98), vi(99)];
        add_pattern(&mut g, &vs1);
        add_pattern(&mut g, &vs2);
        add_pattern(&mut g, &vs3);

        let matches = matcher.find_matches(&g);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].values().copied().collect_vec(), vs1.to_vec());
        assert_eq!(matches[1].values().copied().collect_vec(), vs2.to_vec());
        assert_eq!(matches[2].values().copied().collect_vec(), vs3.to_vec());
    }
}
