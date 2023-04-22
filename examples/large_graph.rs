use std::{fs, path::PathBuf};

use portgraph::PortGraph;
use portmatching::{
    matcher::{
        many_patterns::{LineGraphTrie, ManyPatternMatcher},
        Matcher,
    },
    pattern::Pattern,
};

fn main() {
    let path: PathBuf = ["examples", "data"].iter().collect();
    let mut patterns = Vec::with_capacity(100);
    for i in 0..100 {
        let path = path.join(format!("pattern_{}.bin", i));
        let p: PortGraph = rmp_serde::from_read(fs::File::open(&path).unwrap()).unwrap();
        patterns.push(Pattern::from_graph(p).unwrap());
    }
    let path = path.join("graph_0.bin");
    let graph: PortGraph = rmp_serde::from_read(fs::File::open(&path).unwrap()).unwrap();

    let matcher = LineGraphTrie::from_patterns(patterns.clone());
    // let matcher2 = LineGraphTrie::from_patterns(patterns.clone()).to_cached_trie();

    matcher.find_matches(&graph);
}
