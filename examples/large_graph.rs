use std::{fs, path::PathBuf};

use portgraph::PortGraph;
use portmatching::{matcher::ManyMatcher, Pattern, PortMatcher};

fn main() {
    let path: PathBuf = ["examples", "data"].iter().collect();
    let mut patterns = Vec::with_capacity(100);
    for i in 0..100 {
        let path = path.join(format!("small_circuits/pattern_{}.json", i));
        let p: PortGraph = serde_json::from_reader(fs::File::open(&path).unwrap()).unwrap();
        patterns.push(Pattern::from_portgraph(&p));
    }
    let path = path.join("large_circuits/circuit_0.json");
    let graph: PortGraph = serde_json::from_reader(fs::File::open(path).unwrap()).unwrap();

    println!("Loaded graph and patterns");
    let matcher = ManyMatcher::from_patterns(patterns);
    println!("Built matcher");
    matcher.find_matches(&graph);
}
