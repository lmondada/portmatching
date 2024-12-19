//! Pattern matching for port graphs.

pub mod constraint;
pub mod indexing;
pub mod pattern;
pub mod predicate;
pub mod root_candidates;

pub use constraint::PGConstraint;
pub use pattern::PGPattern;
pub use predicate::PGPredicate;

use crate::{ManyMatcher, NaiveManyMatcher, SinglePatternMatcher};

use portgraph::PortGraph;

use self::indexing::{PGIndexKey, PGIndexingScheme};

/// A matcher for a single port graph pattern.
pub type PGSinglePatternMatcher = SinglePatternMatcher<PGIndexKey, PGPredicate, PGIndexingScheme>;
/// An automaton-based matcher for many port graph patterns.
pub type PGManyPatternMatcher =
    ManyMatcher<PGPattern<PortGraph>, PGIndexKey, PGPredicate, PGIndexingScheme>;
/// A naive matcher for many port graph patterns.
///
/// Use for testing only.
pub type PGNaiveManyPatternMatcher = NaiveManyMatcher<PGIndexKey, PGPredicate, PGIndexingScheme>;

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use crate::{DetHeuristic, HashMap, PatternMatch, PortMatcher};

    use super::*;
    use itertools::Itertools;
    use portgraph::{LinkMut, NodeIndex, PortGraph, PortMut, PortOffset, PortView};
    use rstest::{fixture, rstest};

    #[fixture]
    fn empty_pattern() -> PGPattern<PortGraph> {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let mut p = PGPattern::from_host(g.clone());
        p.pick_root().unwrap();
        p
    }

    #[fixture]
    fn loop_graph() -> PortGraph {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        g
    }

    #[rstest]
    fn empty_pattern_loop_link(empty_pattern: PGPattern<PortGraph>, loop_graph: PortGraph) {
        let matcher = PGSinglePatternMatcher::try_from_pattern(&empty_pattern).unwrap();

        assert_eq!(
            matcher.find_matches(&loop_graph).collect_vec(),
            vec![PatternMatch {
                pattern: 0.into(),
                match_data: HashMap::from_iter([(PGIndexKey::root(0), NodeIndex::new(0))])
            }]
        );
    }

    #[rstest]
    fn empty_pattern_loop_link_many_matcher(
        empty_pattern: PGPattern<PortGraph>,
        loop_graph: PortGraph,
    ) {
        let matcher =
            PGManyPatternMatcher::try_from_patterns(vec![empty_pattern], Default::default())
                .unwrap();

        assert_eq!(
            matcher.find_matches(&loop_graph).collect_vec(),
            vec![PatternMatch {
                pattern: 0.into(),
                match_data: HashMap::from_iter([(PGIndexKey::root(0), NodeIndex::new(0))])
            }]
        );
    }

    #[rstest]
    fn single_pattern_loop_link2(loop_graph: PortGraph) {
        let mut p = PGPattern::from_host(loop_graph.clone());
        p.pick_root().unwrap();
        let matcher = PGSinglePatternMatcher::try_from_pattern(&p).unwrap();

        assert_eq!(
            matcher.find_matches(&loop_graph).collect_vec(),
            vec![PatternMatch {
                pattern: 0.into(),
                match_data: HashMap::from_iter([(PGIndexKey::root(0), NodeIndex::new(0))])
            }]
        );
    }

    #[rstest]
    fn single_pattern_loop_link2_many_matcher(loop_graph: PortGraph) {
        let mut p = PGPattern::from_host(loop_graph.clone());
        p.pick_root().unwrap();
        let matcher = PGManyPatternMatcher::try_from_patterns(vec![p], Default::default()).unwrap();

        assert_eq!(
            matcher.find_matches(&loop_graph).collect_vec(),
            vec![PatternMatch {
                pattern: 0.into(),
                match_data: HashMap::from_iter([(PGIndexKey::root(0), NodeIndex::new(0))])
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = PGPattern::from_host_pick_root(g.clone());
        let matcher = PGManyPatternMatcher::try_from_patterns(vec![p], Default::default()).unwrap();

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n1, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        // We no longer match dangling ports
        // assert_eq!(matcher.find_matches(&g), vec![]);
        assert_eq!(matcher.find_matches(&g).count(), 2);
    }

    #[test]
    fn single_pattern_construction() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(3, 2);
        let n1 = g.add_node(2, 0);
        let n2 = g.add_node(2, 1);
        link(&mut g, (n0, 0), (n0, 2));
        link(&mut g, (n0, 1), (n1, 1));
        link(&mut g, (n2, 0), (n0, 1));
        let p = PGPattern::from_host_pick_root(g.clone());
        let _matcher =
            PGManyPatternMatcher::try_from_patterns(vec![p], Default::default()).unwrap();
    }

    // TODO: weighted graphs
    // #[test]
    // fn one_vertex_pattern() {
    //     let mut g = PortGraph::new();
    //     let n0 = g.add_node(0, 1);
    //     let mut w = UnmanagedDenseMap::new();
    //     w[n0] = 1;
    //     let p = WeightedPattern::from_weighted_portgraph(&g, w);
    //     let m: ManyMatcher<_, _, _> = vec![p].into();

    //     let mut g = PortGraph::new();
    //     let n0 = g.add_node(0, 1);
    //     let n1 = g.add_node(1, 0);
    //     let mut w = UnmanagedDenseMap::new();
    //     w[n0] = 0;
    //     w[n1] = 1;
    //     println!("{}", m.dot_string());
    //     assert_eq!(m.find_matches((&g, &w).into()).len(), 1);
    // }

    fn link<G: LinkMut>(p: &mut G, (n1, p1): (NodeIndex, usize), (n2, p2): (NodeIndex, usize)) {
        p.link_ports(
            p.port_index(n1, PortOffset::new_outgoing(p1)).unwrap(),
            p.port_index(n2, PortOffset::new_incoming(p2)).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn two_simple_patterns() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(0, 1);
        let n1 = p1.add_node(2, 0);
        let n2 = p1.add_node(0, 1);
        link(&mut p1, (n0, 0), (n1, 0));
        link(&mut p1, (n2, 0), (n1, 1));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 2);
        let n1 = g.add_node(3, 1);
        let n2 = g.add_node(0, 2);
        let n3 = g.add_node(3, 2);
        link(&mut g, (n0, 0), (n1, 2));
        link(&mut g, (n0, 1), (n3, 0));
        link(&mut g, (n1, 0), (n3, 1));
        link(&mut g, (n2, 0), (n1, 0));
        link(&mut g, (n2, 1), (n3, 2));
        link(&mut g, (n3, 0), (n1, 1));
        link(&mut g, (n3, 1), (n0, 0));

        let p1 = PGPattern::from_host_with_root(p1, n0);
        let p2 = PGPattern::from_host_with_root(p2, n0);
        let matcher =
            PGManyPatternMatcher::try_from_patterns(vec![p1, p2], Default::default()).unwrap();
        assert_eq!(matcher.find_matches(&g).count(), 3);
    }

    #[test]
    fn two_simple_patterns_change_root() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(0, 1);
        let n1 = p1.add_node(2, 0);
        let n2 = p1.add_node(0, 1);
        link(&mut p1, (n0, 0), (n1, 0));
        link(&mut p1, (n2, 0), (n1, 1));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 2);
        let n1 = g.add_node(3, 1);
        let n2 = g.add_node(0, 2);
        let n3 = g.add_node(3, 2);
        link(&mut g, (n0, 0), (n1, 2));
        link(&mut g, (n0, 1), (n3, 0));
        link(&mut g, (n1, 0), (n3, 1));
        link(&mut g, (n2, 0), (n1, 0));
        link(&mut g, (n2, 1), (n3, 2));
        link(&mut g, (n3, 0), (n1, 1));
        link(&mut g, (n3, 1), (n0, 0));

        let p1 = PGPattern::from_host_with_root(p1, n1);
        let p2 = PGPattern::from_host_with_root(p2, n0);
        let mut rd_cnt = 0;
        let matcher = PGManyPatternMatcher::try_from_patterns_with_det_heuristic(
            vec![p1, p2],
            Default::default(),
            DetHeuristic::Custom(RefCell::new(Box::new(move |_| {
                rd_cnt += 1;
                rd_cnt <= 3
            }))),
        )
        .unwrap();
        assert_eq!(matcher.find_matches(&g).count(), 3);
    }

    #[test]
    fn trie_construction_fail() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(2, 1);
        let n1 = p1.add_node(1, 0);
        link(&mut p1, (n0, 0), (n1, 0));
        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 0);
        let n1 = p2.add_node(0, 2);
        link(&mut p2, (n1, 0), (n0, 1));
        link(&mut p2, (n1, 1), (n0, 0));
        PGManyPatternMatcher::try_from_patterns(
            [p1, p2].map(PGPattern::from_host_pick_root).to_vec(),
            Default::default(),
        )
        .unwrap();
    }

    #[test]
    fn two_simple_patterns_2() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(1, 0);
        let n1 = p1.add_node(1, 3);
        let n2 = p1.add_node(0, 3);
        let n3 = p1.add_node(1, 3);
        link(&mut p1, (n1, 0), (n0, 0));
        link(&mut p1, (n2, 0), (n1, 0));
        link(&mut p1, (n2, 2), (n3, 0));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let p1 = PGPattern::from_host_pick_root(p1);
        let p2 = PGPattern::from_host_pick_root(p2);
        let matcher =
            PGManyPatternMatcher::try_from_patterns(vec![p1, p2], Default::default()).unwrap();
        assert_eq!(matcher.find_matches(&g).count(), 1);
    }

    #[test]
    fn two_simple_patterns_2_non_det() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(1, 0);
        let n1 = p1.add_node(1, 3);
        let n2 = p1.add_node(0, 3);
        let n3 = p1.add_node(1, 3);
        link(&mut p1, (n1, 0), (n0, 0));
        link(&mut p1, (n2, 0), (n1, 0));
        link(&mut p1, (n2, 2), (n3, 0));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let p1 = PGPattern::from_host_pick_root(p1.clone());
        let p2 = PGPattern::from_host_pick_root(p2);
        let matcher = PGManyPatternMatcher::try_from_patterns_with_det_heuristic(
            vec![p1, p2],
            Default::default(),
            DetHeuristic::Never,
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g).count(), 1);
    }

    #[test]
    fn two_simple_patterns_3() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(3, 1);
        let n1 = p1.add_node(0, 3);
        link(&mut p1, (n1, 1), (n0, 0));
        link(&mut p1, (n0, 0), (n0, 2));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(0, 2);
        let n1 = p2.add_node(2, 0);
        link(&mut p2, (n0, 0), (n1, 1));
        link(&mut p2, (n0, 1), (n1, 0));

        let p1 = PGPattern::from_host_pick_root(p1);
        let p2 = PGPattern::from_host_pick_root(p2);
        PGManyPatternMatcher::try_from_patterns(vec![p1, p2], Default::default()).unwrap();
    }

    #[test]
    fn single_pattern_single_node() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = PGPattern::from_host_pick_root(g);
        let matcher = PGSinglePatternMatcher::try_from_pattern(&p).unwrap();
        let mut g = PortGraph::new();
        g.add_node(1, 0);

        assert_eq!(matcher.find_matches(&g).count(), 1);
    }

    // TODO: weighted graphs
    // #[test]
    // fn single_pattern_single_node_weighted() {
    //     let mut g = PortGraph::new();
    //     let n0 = g.add_node(0, 1);
    //     let mut w = UnmanagedDenseMap::new();
    //     w[n0] = 1;
    //     let p = WeightedPGPattern::from_weighted_portgraph(&g, w);
    //     let m = SinglePatternMatcher::from_constraints(p);

    //     let mut g = PortGraph::new();
    //     let n0 = g.add_node(0, 1);
    //     let n1 = g.add_node(1, 0);
    //     let mut w = UnmanagedDenseMap::new();
    //     w[n0] = 0;
    //     w[n1] = 1;
    //     assert_eq!(m.find_matches((&g, &w).into()).len(), 1);
    // }

    #[test]
    fn single_node_loop() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();
        let p = PGPattern::from_host_pick_root(g);
        let matcher = PGSinglePatternMatcher::try_from_pattern(&p).unwrap();

        let mut g = PortGraph::new();
        let n = g.add_node(2, 1);
        g.link_ports(
            g.port_index(n, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g).count(), 1);
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
        let p = PGPattern::from_host_pick_root(g);
        let matcher = PGSinglePatternMatcher::try_from_pattern(&p).unwrap();

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port_index(n0, PortOffset::new_outgoing(0)).unwrap(),
            g.port_index(n1, PortOffset::new_incoming(0)).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g).count(), 0);
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
        let p = PGPattern::from_host_pick_root(pattern);
        let matcher = PGSinglePatternMatcher::try_from_pattern(&p).unwrap();

        let mut g = PortGraph::new();
        for _ in 0..100 {
            g.add_node(3, 3);
        }
        let vi = |i| g.nodes_iter().nth(i).unwrap();
        let vs1 = [vi(0), vi(10), vi(30), vi(55)];
        let vs2 = [vi(3), vi(12), vi(23), vi(44)];
        let vs3 = [vi(12), vi(55), vi(98), vi(99)];
        add_pattern(&mut g, &vs1);
        add_pattern(&mut g, &vs2);
        add_pattern(&mut g, &vs3);

        let mut matches = matcher
            .find_matches(&g)
            .into_iter()
            .map(|m| {
                m.match_data
                    .values()
                    .copied()
                    .sorted()
                    .dedup()
                    .collect_vec()
            })
            .collect_vec();
        matches.sort_unstable_by_key(|v| *v.first().unwrap());
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], vs1.to_vec());
        assert_eq!(matches[1], vs2.to_vec());
        assert_eq!(matches[2], vs3.to_vec());
    }
}
