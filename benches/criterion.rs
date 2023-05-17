use std::fs::File;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use itertools::Itertools;

use portgraph::PortGraph;
use portmatching::matcher::many_patterns::{ManyPatternMatcher, NaiveManyMatcher, TrieMatcher};
use portmatching::matcher::Matcher;
use portmatching::pattern::UnweightedPattern;
use portmatching::TrieConstruction;

fn bench_matching<M: Matcher>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[UnweightedPattern],
    sizes: impl Iterator<Item = usize>,
    graph: &PortGraph,
    mut get_matcher: impl FnMut(Vec<UnweightedPattern>) -> M,
) {
    group.sample_size(10);
    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            let matcher = get_matcher(patterns);
            b.iter(|| criterion::black_box(matcher.find_matches(graph)));
        });
    }
}

fn bench_trie_construction<M: Matcher>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[UnweightedPattern],
    sizes: impl Iterator<Item = usize>,
    mut get_matcher: impl FnMut(Vec<UnweightedPattern>) -> M,
) {
    group.sample_size(10);
    for n in sizes {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            b.iter(|| criterion::black_box(get_matcher(patterns.clone())));
            // let matcher = get_matcher(patterns);
            // b.iter(|| matcher.find_matches(graph));
        });
    }
}

fn perform_benches(c: &mut Criterion) {
    let patterns = glob::glob("datasets/small_circuits/*.json")
        .expect("cannot read small circuits directory")
        .map(|p| {
            let g = serde_json::from_reader(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small circuit"),
            )
            .expect("could not serialize");
            UnweightedPattern::from_graph(g).expect("pattern not connected")
        })
        .collect_vec();
    // TODO: use more than one graph in benchmark
    let graph = glob::glob("datasets/large_circuits/*.json")
        .expect("cannot read large circuits directory")
        .map(|p| {
            serde_json::from_reader(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small circuit"),
            )
            .expect("could not serialize")
        })
        .next()
        .expect("Did not find any large circuit");

    let mut group = c.benchmark_group("Many Patterns Matching");
    bench_matching(
        "Naive matching",
        &mut group,
        &patterns,
        (0..=300).step_by(100),
        &graph,
        NaiveManyMatcher::from_patterns,
    );
    bench_matching(
        "Balanced Graph Trie",
        &mut group,
        &patterns,
        (0..=1000).step_by(100),
        &graph,
        TrieMatcher::from_patterns,
    );
    bench_matching(
        "Non-deterministic Graph Trie",
        &mut group,
        &patterns,
        (0..=1000).step_by(100),
        &graph,
        |ps| {
            let mut matcher = TrieMatcher::new(TrieConstruction::NonDeterministic);
            for p in ps {
                matcher.add_pattern(p);
            }
            matcher
        },
    );
    bench_matching(
        "Deterministic Graph Trie",
        &mut group,
        &patterns,
        (0..=300).step_by(100),
        &graph,
        |ps| {
            let mut matcher = TrieMatcher::new(TrieConstruction::Deterministic);
            for p in ps {
                matcher.add_pattern(p);
            }
            matcher
        },
    );
    // This is too slow
    // bench_matching("Naive", &mut group, &patterns, &graph, |p| {
    //     NaiveManyMatcher::from_patterns(p)
    // });
    group.finish();

    let mut group = c.benchmark_group("Trie Construction");
    bench_trie_construction(
        "Balanced Graph Trie",
        &mut group,
        &patterns,
        (0..=300).step_by(30),
        TrieMatcher::from_patterns,
    );
    bench_trie_construction(
        "Non-deterministic Graph Trie",
        &mut group,
        &patterns,
        (0..=300).step_by(30),
        |ps| {
            let mut matcher = TrieMatcher::new(TrieConstruction::NonDeterministic);
            for p in ps {
                matcher.add_pattern(p);
            }
            matcher
        },
    );
    bench_trie_construction(
        "Deterministic Graph Trie",
        &mut group,
        &patterns,
        (0..=300).step_by(30),
        |ps| {
            let mut matcher = TrieMatcher::new(TrieConstruction::Deterministic);
            for p in ps {
                matcher.add_pattern(p);
            }
            matcher
        },
    );
    group.finish();
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
