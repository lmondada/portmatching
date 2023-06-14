use std::{fs, io, time::Instant};

use portgraph::PortGraph;
use portmatching::{
    constraint::{Address, UnweightedAdjConstraint},
    pattern::UnweightedPattern,
    Matcher, TrieMatcher,
};

fn main() {
    let mut wtr = csv::Writer::from_writer(io::stdout());
    let file_name = format!("examples/data/large_circuits/circuit_0.json");
    let graph: PortGraph = serde_json::from_reader(fs::File::open(file_name).unwrap()).unwrap();
    println!("Loaded graph and patterns");
    let prefix = "balanced";
    for size in (500..=2000).step_by(500) {
        println!("Loading trie for size {size}...");
        let t = Instant::now();
        let file_name = format!("datasets/xxl/tries/{prefix}_{size}.bin");
        let mut matcher: TrieMatcher<UnweightedAdjConstraint, Address, UnweightedPattern> =
            rmp_serde::from_read(fs::File::open(file_name).unwrap())
                .expect("could not deserialize trie");
        println!("Done in {:?}secs.", t.elapsed().as_secs());

        println!("Optimising...");
        let t = Instant::now();
        matcher.optimise(5, 5);
        println!("Done in {:?}secs.", t.elapsed().as_secs());

        // some node in the middle
        let mut sum_trace = [0; 30];
        for node in graph.nodes_iter() {
            let trace = matcher.find_anchored_matches((&graph, node));
            for (i, t) in trace.into_iter().enumerate() {
                sum_trace[i] += t;
            }
        }
        wtr.write_record(
            [
                vec![size.to_string()],
                sum_trace.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            ]
            .concat(),
        )
        .unwrap();
    }
    // matcher.optimise();
    // println!("Number of trie states: {}", matcher.n_states());
    // println!("Number of ports (capacity) after optimisation: {}", matcher.port_capacity());
    // println!("Number of ports after optimisation: {}", matcher.n_ports());
}
