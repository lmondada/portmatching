use std::{fs, path::PathBuf, hint::black_box};

use portgraph::PortGraph;
use portmatching::{matcher::UnweightedManyMatcher, PortMatcher};

fn main() {
    let path: PathBuf = ["datasets", "xxl"].iter().collect();

    let file_name = path.join(&format!("tries/balanced_1000.bin"));
    let matcher: UnweightedManyMatcher = rmp_serde::from_read(fs::File::open(file_name).unwrap())
        .expect("could not deserialize trie");

    let path = path.join("large_circuits/circuit_0.json");
    let graph: PortGraph = serde_json::from_reader(fs::File::open(path).unwrap()).unwrap();

    println!("Loaded graph and patterns");
    for _ in 0..5 {
        black_box(matcher.find_matches(&graph));
    }
}
