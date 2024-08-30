use std::{collections::HashSet, fs};

use glob::glob;
use itertools::Itertools;
use portgraph::proptest::gen_portgraph;
use proptest::prelude::*;

use portmatching::{
    portgraph::{PGManyPatternMatcher, PGNaiveManyPatternMatcher, PGPattern},
    utils::{gen_portgraph_connected, test::SerialPatternMatch},
    PortMatcher,
};

const DBG_DUMP_FILES: bool = false;

proptest! {
    #[ignore = "a bit slow"]
    #[test]
    fn many_graphs_proptest(
        pattern_graphs in prop::collection::vec(gen_portgraph_connected(6, 4, 20), 1..10),
        g in gen_portgraph(30, 4, 60)
    ) {
        if DBG_DUMP_FILES {
            for path in glob("pattern_*.json").expect("glob pattern failed").flatten() {
                fs::remove_file(path).expect("Removing file failed");
            }
            fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
        }
        let patterns = pattern_graphs
            .iter()
            .cloned()
            .map(PGPattern::from_host_pick_root)
            .collect_vec();
        if DBG_DUMP_FILES {
            for (i, p) in patterns.iter().enumerate() {
                fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(&p).unwrap()).unwrap();
            }
        }
        let naive = PGNaiveManyPatternMatcher::try_from_patterns(&patterns).unwrap();
        let single_matches: HashSet<SerialPatternMatch>  = naive.find_matches(&g).map_into().collect();
        if DBG_DUMP_FILES {
            fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
        }
        let matcher = PGManyPatternMatcher::try_from_patterns(patterns, Default::default()).unwrap();
        let many_matches: HashSet<SerialPatternMatch> = matcher.find_matches(&g).map_into().collect();
        prop_assert_eq!(many_matches, single_matches);
    }
}

proptest! {
    #[ignore = "a bit slow"]
    #[test]
    fn many_graphs_proptest_small(
        pattern_graphs in prop::collection::vec(gen_portgraph_connected(4, 4, 20), 1..5),
        g in gen_portgraph(30, 4, 60)
    ) {
        if DBG_DUMP_FILES {
            for path in glob("pattern_*.json").expect("glob pattern failed").flatten() {
                fs::remove_file(path).expect("Removing file failed");
            }
            fs::write("graph.json", serde_json::to_vec(&g).unwrap()).unwrap();
            println!("==== graph.json ====");
            println!("{}", serde_json::to_string(&g).unwrap());
        }
        let patterns = pattern_graphs
            .iter()
            .cloned()
            .map(PGPattern::from_host_pick_root)
            .collect_vec();
        if DBG_DUMP_FILES {
            for (i, p) in patterns.iter().enumerate() {
                fs::write(&format!("pattern_{}.json", i), serde_json::to_vec(&p).unwrap()).unwrap();
                println!("==== pattern_{}.json ====", i);
                println!("{}", serde_json::to_string(&p).unwrap());
            }
        }
        let naive = PGNaiveManyPatternMatcher::try_from_patterns(&patterns).unwrap();
        let single_matches: HashSet<SerialPatternMatch> = naive.find_matches(&g).map_into().collect();
        if DBG_DUMP_FILES {
            fs::write("results.json", serde_json::to_vec(&single_matches).unwrap()).unwrap();
            println!("==== results.json ====");
            println!("{}", serde_json::to_string(&single_matches).unwrap());
        }
        let matcher = PGManyPatternMatcher::try_from_patterns(patterns, Default::default()).unwrap();
        let many_matches: HashSet<SerialPatternMatch> = matcher.find_matches(&g).map_into().collect();
        assert_eq!(many_matches, single_matches);
        prop_assert_eq!(many_matches, single_matches);
    }
}
