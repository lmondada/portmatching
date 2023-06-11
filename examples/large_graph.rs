use std::{fs, path::PathBuf};

use portgraph::PortGraph;
use portmatching::{pattern::UnweightedPattern, ManyPatternMatcher, Matcher, TrieMatcher};

fn main() {
    let path: PathBuf = ["examples", "data"].iter().collect();
    let mut patterns = Vec::with_capacity(100);
    for i in 0..100 {
        let path = path.join(format!("small_graphs/pattern_{}.json", i));
        let p: PortGraph = serde_json::from_reader(fs::File::open(&path).unwrap()).unwrap();
        patterns.push(UnweightedPattern::from_graph(p).unwrap());
    }
    let path = path.join("large_graphs/graph_0.json");
    let graph: PortGraph = serde_json::from_reader(fs::File::open(path).unwrap()).unwrap();

    println!("Loaded graph and patterns");
    let matcher = TrieMatcher::from_patterns(patterns);
    println!("Built matcher");
    matcher.find_matches(&graph);
}
