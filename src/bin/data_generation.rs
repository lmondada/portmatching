use clap::Parser;
use portmatching::{
    matcher::UnweightedManyMatcher, utils::is_connected, Pattern, UnweightedPattern,
};
use std::{
    cmp,
    fs::{self, File},
};

use portgraph::{LinkMut, LinkView, NodeIndex, PortGraph, PortMut, PortView, UnmanagedDenseMap};
use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Number of large circuits
    #[arg(short = 'K')]
    #[arg(default_value_t = 100)]
    n_large: usize,
    // Number of small circuits,
    #[arg(short = 'k')]
    #[arg(default_value_t = 100)]
    n_small: usize,

    // Max gates in large circuits
    #[arg(short = 'N')]
    #[arg(default_value_t = 200)]
    v_large: usize,
    // Max gates in small circuits
    #[arg(short = 'n')]
    #[arg(default_value_t = 10)]
    v_small: usize,

    // Max qubits in large circuits
    #[arg(short = 'Q')]
    #[arg(default_value_t = 400)]
    q_large: usize,
    // Max qubits in small circuits
    #[arg(short = 'q')]
    #[arg(default_value_t = 5)]
    q_small: usize,

    // Max in/out-degree in large circuits
    #[arg(short = 'D')]
    #[arg(default_value_t = 5)]
    d_large: usize,
    // Max in/out-degree in small circuits
    #[arg(short = 'd')]
    #[arg(default_value_t = 5)]
    d_small: usize,

    // Whether to precompile trie
    #[arg(long)]
    #[arg(short = 'X')]
    pre_compile: Option<String>,

    // Circuits location
    #[arg(short = 'o')]
    #[arg(default_value = "datasets")]
    directory: String,
}

fn main() {
    let args = Args::parse();

    let mut rng = StdRng::seed_from_u64(1234);

    let dir = args.directory;

    if let Some(sizes) = args.pre_compile {
        let pattern_path = format!("{dir}/small_circuits/pattern_*.json");
        let (patterns, _weighted_patterns): (Vec<_>, Vec<_>) = glob::glob(&pattern_path)
            .expect("cannot read small circuits directory")
            .map(|p| {
                let g: PortGraph = serde_json::from_reader(
                    File::open(p.as_ref().expect("path does not exist?"))
                        .expect("Could not open small circuit"),
                )
                .expect("could not serialize");
                let _w = gen_weights(g.nodes_iter());
                (
                    Pattern::from_portgraph(&g),
                    (), // Pattern::from_weighted_graph(g, w).expect("pattern not connected"),
                )
            })
            .unzip();
        let sizes = sizes
            .split(',')
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        precompile(&patterns, &sizes, &dir);
        // precompile_weighted(&weighted_patterns, &sizes, &dir);
    } else {
        // large circuits
        {
            fs::create_dir_all(format!("{dir}/large_circuits"))
                .expect("could not create directory");
            let (n_circuits, n, q, d) = (args.n_large, args.v_large, args.q_large, args.d_large);
            for i in 0..n_circuits {
                println!("{}/{n_circuits} large circuits...", i + 1);
                let g = gen_circ(n, q, d, |c| !exists_two_cx_gates(c), &mut rng);
                let f = format!("{dir}/large_circuits/circuit_{i}.json");
                fs::write(f, serde_json::to_vec(&g).unwrap()).expect("could not write to file");
            }
        }
        // small circuits
        {
            fs::create_dir_all(format!("{dir}/small_circuits"))
                .expect("could not create directory");
            let (n_circuits, n, q, d) = (args.n_small, args.v_small, args.q_small, args.d_small);
            for i in 0..n_circuits {
                if i % 100 == 99 {
                    println!("{}/{n_circuits} small circuits...", i + 1);
                }
                let g = gen_circ(
                    n,
                    q,
                    d,
                    |c| is_connected(c) && exists_two_cx_gates(c),
                    &mut rng,
                );
                let f = format!("{dir}/small_circuits/pattern_{i}.json");
                fs::write(f, serde_json::to_vec(&g).unwrap()).expect("could not write to file");
                Pattern::from_portgraph(&g);
            }
        };
    }
}

fn gen_weights(nodes: impl Iterator<Item = NodeIndex>) -> UnmanagedDenseMap<NodeIndex, usize> {
    let mut rng = rand::thread_rng();
    let mut weights = UnmanagedDenseMap::new();
    for n in nodes {
        weights[n] = rng.gen_range(0..8);
    }
    weights
}

fn precompile(patterns: &[UnweightedPattern], sizes: &[usize], dir: &str) {
    fs::create_dir_all(format!("{dir}/tries")).expect("could not create directory");

    let n_sizes = sizes.len();
    for (i, &l) in sizes.iter().enumerate() {
        println!("Compiling size {l}... ({}/{n_sizes})", i + 1);
        let matcher = UnweightedManyMatcher::from_patterns(patterns[..l].to_vec());
        fs::write(
            format!("{dir}/tries/balanced_{l}.bin"),
            rmp_serde::to_vec(&matcher).unwrap(),
        )
        .unwrap_or_else(|_| panic!("could not write to {dir}/tries"));
        println!("Optimising size {l}... ({}/{n_sizes})", i + 1);
    }
}

/*fn precompile_weighted<N: Ord + Clone + serde::Serialize + 'static>(
    patterns: &[WeightedPattern<N>],
    sizes: &[usize],
    dir: &str,
) {
    let size_cutoff = 5;
    let depth_cutoff = 8;
    fs::create_dir_all(format!("{dir}/tries")).expect("could not create directory");

    let n_sizes = sizes.len();
    let mut last_size = 0;
    let mut matcher = TrieMatcher::default();
    for (i, &l) in sizes.iter().enumerate() {
        assert!(l > last_size);
        println!("Compiling size {l}... ({}/{n_sizes})", i + 1);
        for p in &patterns[last_size..l] {
            matcher.add_pattern(p.clone());
        }
        fs::write(
            format!("{dir}/tries/weighted_balanced_{l}.bin"),
            rmp_serde::to_vec(&matcher).unwrap(),
        )
        .unwrap_or_else(|_| panic!("could not write to {dir}/tries"));
        println!("Optimising size {l}... ({}/{n_sizes})", i + 1);
        let mut opt_matcher = matcher.clone();
        opt_matcher.optimise(size_cutoff, depth_cutoff);
        fs::write(
            format!("{dir}/tries/weighted_optimised_{l}.bin"),
            rmp_serde::to_vec(&opt_matcher).unwrap(),
        )
        .unwrap_or_else(|_| panic!("could not write to {dir}/tries"));
        last_size = l;
    }
}*/

fn gen_qubits<R: Rng>(q: usize, k: usize, rng: &mut R) -> Vec<usize> {
    let qubits = Uniform::from(0..q);
    let mut ret = Vec::with_capacity(k);
    assert!(k <= q);
    for _ in 0..k {
        let mut q = None;
        while q.is_none() || ret.contains(q.as_ref().unwrap()) {
            q = Some(qubits.sample(rng));
        }
        ret.push(q.unwrap());
    }
    ret
}

/// Generate a random circuit
///
/// # Arguments:
///  * n: number of gates
///  * q: number of qubits
///  * d: max in- and out-degree
///  * pred: a predicate the circuit must satisfy
///  * rng: a random number generator
///
/// The predicate should be montonic, i.e. if it returns true for a circuit, it
/// should also return true for any circuit with more gates.
fn gen_circ<R, F>(n: usize, q: usize, d: usize, pred: F, rng: &mut R) -> PortGraph
where
    R: Rng,
    F: Fn(&PortGraph) -> bool,
{
    let mut circ = None;
    let mut n_fails = 0;
    let d = cmp::min(d, q);
    let arity = Uniform::from(1..=d);
    while circ.is_none() && n_fails < 10000 {
        circ = {
            let mut prev_gates = vec![None; q];
            let mut circ = PortGraph::new();
            for _ in 0..n {
                let k = arity.sample(rng);
                let n = circ.add_node(k, k);
                let qbs = gen_qubits(q, k, rng);
                for (&q, in_p) in qbs.iter().zip(circ.inputs(n)) {
                    if let Some(prev_out) = prev_gates[q] {
                        circ.link_ports(prev_out, in_p).expect("could not link");
                    }
                }
                for (&q, out_p) in qbs.iter().zip(circ.outputs(n)) {
                    prev_gates[q] = out_p.into();
                }
            }
            pred(&circ).then_some(circ)
        };
        n_fails += 1;
    }
    circ.expect("could not generate circuit")
}

/// Whether there are two successive 2qb gates on the same two qubits
fn exists_two_cx_gates(circ: &PortGraph) -> bool {
    for n in circ.nodes_iter() {
        if circ.num_outputs(n) != 2 {
            continue;
        }
        let mut next = circ.outputs(n).map(|out| circ.port_link(out));
        let Some(next0) = next.next().unwrap() else { continue };
        let Some(next1) = next.next().unwrap() else { continue };
        let next = [next0, next1];
        if circ.port_offset(next[0]).expect("invalid port").index() != 0 {
            continue;
        }
        if circ.port_offset(next[1]).expect("invalid port").index() != 1 {
            continue;
        }
        if circ.port_node(next[0]) != circ.port_node(next[1]) {
            continue;
        }
        let next = circ.port_node(next[0]).unwrap();
        if circ.num_inputs(next) != 2 {
            continue;
        }
        return true;
    }
    false
}
