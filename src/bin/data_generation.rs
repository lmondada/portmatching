use clap::Parser;
use portmatching::{
    utils::is_connected, BalancedTrieMatcher, DetTrieMatcher, ManyPatternMatcher, NaiveManyMatcher,
    NonDetTrieMatcher, Pattern,
};
use rmp_serde;
use serde::Serialize;
use std::fs;

use portgraph::{Direction, PortGraph, PortIndex};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Number of large graphs
    #[arg(short = 'K')]
    #[arg(default_value_t = 100)]
    n_large: usize,
    // Number of small graphs,
    #[arg(short = 'k')]
    #[arg(default_value_t = 100)]
    n_small: usize,

    // Max vertices in large graph
    #[arg(short = 'N')]
    #[arg(default_value_t = 200)]
    v_large: usize,
    // Max vertices in small graph
    #[arg(short = 'n')]
    #[arg(default_value_t = 10)]
    v_small: usize,

    // Max edges in large graph
    #[arg(short = 'E')]
    #[arg(default_value_t = 400)]
    e_large: usize,
    // Max edges in small graph
    #[arg(short = 'e')]
    #[arg(default_value_t = 40)]
    e_small: usize,

    // Max in/out-degree in large graph
    #[arg(short = 'D')]
    #[arg(default_value_t = 5)]
    d_large: usize,
    // Max in/out-degree in small graph
    #[arg(short = 'd')]
    #[arg(default_value_t = 5)]
    d_small: usize,

    // Graph location
    #[arg(short = 'o')]
    #[arg(default_value = "datasets")]
    directory: String,
}

fn main() {
    let args = Args::parse();

    let mut rng = StdRng::seed_from_u64(1234);

    // large graphs
    let dir = args.directory;
    {
        fs::create_dir_all(format!("{dir}/large_graphs")).expect("could not create directory");
        let (n_graphs, n, m, d) = (args.n_large, args.v_large, args.e_large, args.d_large);
        for i in 0..n_graphs {
            println!("{}/{n_graphs} large graphs...", i + 1);
            let g = gen_graph(n, m, d, &mut rng).expect("could not generate graph");
            let f = format!("{dir}/large_graphs/graph_{i}.bin");
            fs::write(f, rmp_serde::to_vec(&g).unwrap()).expect("could not write to file");
        }
    }
    // small graphs
    let patterns = {
        fs::create_dir_all(format!("{dir}/small_graphs")).expect("could not create directory");
        let (n_graphs, n, m, d) = (args.n_small, args.v_small, args.e_small, args.d_small);
        (0..n_graphs)
            .map(|i| {
                println!("{}/{n_graphs} small graphs...", i + 1);
                let mut g = None;
                let mut n_fails = 0;
                while g.is_none() || !is_connected(g.as_ref().unwrap()) {
                    g = gen_graph(n, m, d, &mut rng);
                    n_fails += 1;
                    if n_fails >= 10000 {
                        panic!("could not create connected graph with n={n}, m={m}, d={d}")
                    }
                }
                let g = g.unwrap();
                let f = format!("{dir}/small_graphs/pattern_{i}.bin");
                fs::write(f, rmp_serde::to_vec(&g).unwrap()).expect("could not write to file");
                g
            })
            .collect::<Vec<_>>()
    };

    println!("Precompute graph tries");
    let dir = format!("{dir}/tries");
    fs::create_dir_all(&dir).expect("could not create directory");
    construct_tries(patterns, &dir);
}

fn construct_tries(patterns: Vec<PortGraph>, dir: &str) {
    let patterns = patterns
        .into_iter()
        .map(|g| Pattern::from_graph(g).unwrap())
        .collect::<Vec<_>>();
    construct_trie(
        BalancedTrieMatcher::from_patterns,
        &patterns,
        "balanced_trie",
        dir,
    );
    construct_trie(
        |ps| BalancedTrieMatcher::from_patterns(ps).to_cached_trie(),
        &patterns,
        "balanced_trie_cached",
        dir,
    );
    construct_trie(DetTrieMatcher::from_patterns, &patterns, "det_trie", dir);
    construct_trie(
        NonDetTrieMatcher::from_patterns,
        &patterns,
        "non_det_trie",
        dir,
    );
    construct_trie(
        NaiveManyMatcher::from_patterns,
        &patterns,
        "naive_matcher",
        dir,
    );
}

fn construct_trie<F: Fn(Vec<Pattern>) -> T, T: Serialize>(
    builder: F,
    patterns: &Vec<Pattern>,
    name: &str,
    dir: &str,
) {
    for i in (0..patterns.len()).step_by(100) {
        let patterns = patterns[0..i].to_vec();
        println!("Constructing {} for {} patterns", name, i);
        let matcher = builder(patterns);
        let f = format!("{dir}/{name}_{i}.bin");
        fs::write(f, rmp_serde::to_vec(&matcher).unwrap()).expect("could not write to file");
    }
}

/// Generate a random graph
///
/// # Arguments:
///  * n: number of vertices
///  * m: number of edges
///  * d: max in- and out-degree
fn gen_graph<R: Rng>(n: usize, m: usize, d: usize, rng: &mut R) -> Option<PortGraph> {
    let in_degrees = gen_degrees(n, m, d, rng)?;
    let out_degrees = gen_degrees(n, m, d, rng)?;

    let mut g = PortGraph::new();
    for (i, o) in in_degrees.into_iter().zip(out_degrees) {
        g.add_node(i, o);
    }
    let max_port = g.port_count();
    for _ in 0..m {
        let in_p = gen_port(max_port, &g, Direction::Incoming, rng)?;
        let out_p = gen_port(max_port, &g, Direction::Outgoing, rng)?;
        g.link_ports(out_p, in_p).expect("could not link");
    }
    Some(g)
}

fn gen_port<R: Rng>(
    max_port: usize,
    g: &PortGraph,
    dir: Direction,
    rng: &mut R,
) -> Option<PortIndex> {
    let port = Uniform::from(0..max_port).map(|p| PortIndex::new(p));
    let mut p = port.sample(rng);
    let mut n_fails = 0;
    while g.port_direction(p).expect("invalid port") != dir || g.port_link(p).is_some() {
        p = port.sample(rng);
        n_fails += 1;
        if n_fails >= 10000 {
            return None;
        }
    }
    Some(p)
}

/// Generate a vec of size n, with values in 0..=d and sum >= m
fn gen_degrees<R: Rng>(n: usize, m: usize, d: usize, rng: &mut R) -> Option<Vec<usize>> {
    let mut vec = Vec::new();
    let mut n_fails = 0;
    let degree = Uniform::from(0..=d);
    while vec.iter().copied().sum::<usize>() < m {
        vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(degree.sample(rng));
        }
        n_fails += 1;
        if n_fails >= 10000 {
            return None;
        }
    }
    Some(vec)
}
