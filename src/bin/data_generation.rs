use clap::Parser;
use portmatching::utils::is_connected;
use rmp_serde;
use std::{cmp, fs};

use portgraph::PortGraph;
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

    // Circuits location
    #[arg(short = 'o')]
    #[arg(default_value = "datasets")]
    directory: String,
}

fn main() {
    let args = Args::parse();

    let mut rng = StdRng::seed_from_u64(1234);

    // large circuits
    let dir = args.directory;
    {
        fs::create_dir_all(format!("{dir}/large_circuits")).expect("could not create directory");
        let (n_circuits, n, q, d) = (args.n_large, args.v_large, args.q_large, args.d_large);
        for i in 0..n_circuits {
            println!("{}/{n_circuits} large circuits...", i + 1);
            let g = gen_circ(n, q, d, &mut rng);
            let f = format!("{dir}/large_circuits/circuit_{i}.bin");
            fs::write(f, rmp_serde::to_vec(&g).unwrap()).expect("could not write to file");
        }
    }
    // small circuits
    {
        fs::create_dir_all(format!("{dir}/small_circuits")).expect("could not create directory");
        let (n_circuits, n, q, d) = (args.n_small, args.v_small, args.q_small, args.d_small);
        for i in 0..n_circuits {
            println!("{}/{n_circuits} small circuits...", i + 1);
            let mut g = None;
            let mut n_fails = 0;
            while g.is_none() || !is_connected(g.as_ref().unwrap()) {
                g = Some(gen_circ(n, q, d, &mut rng));
                n_fails += 1;
                if n_fails >= 1000 {
                    panic!("could not create connected circuit with n={n}, q={q}, d={d}")
                }
            }
            let g = g.unwrap();
            let f = format!("{dir}/small_circuits/pattern_{i}.bin");
            fs::write(f, rmp_serde::to_vec(&g).unwrap()).expect("could not write to file");
        }
    }
}

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
fn gen_circ<R: Rng>(n: usize, q: usize, d: usize, rng: &mut R) -> PortGraph {
    let mut prev_gates = vec![None; q];
    let d = cmp::min(d, q);
    let arity = Uniform::from(1..=d);
    let mut circ = PortGraph::new();
    for _ in 0..n {
        let k = arity.sample(rng);
        let v = circ.add_node(k, k);
        for (i, q) in gen_qubits(q, k, rng).into_iter().enumerate() {
            let in_p = circ.input(v, i).unwrap();
            let out_p = circ.output(v, i).unwrap();
            if let Some(prev_out) = prev_gates[q] {
                circ.link_ports(prev_out, in_p).expect("could not link");
            }
            prev_gates[q] = Some(out_p);
        }
    }
    circ
}
