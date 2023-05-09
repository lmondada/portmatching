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
use portmatching::graph_tries::BaseGraphTrie;
use portmatching::matcher::many_patterns::{
    BalancedTrieMatcher,
    DetTrieMatcher,
    ManyPatternMatcher,
    NonDetTrieMatcher,
    // NaiveManyMatcher,
};
use portmatching::matcher::Matcher;
use portmatching::pattern::Pattern;

fn bench_matching<M: Matcher, F: FnMut(Vec<Pattern>) -> M>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern],
    graph: &PortGraph,
    mut get_matcher: F,
) {
    group.sample_size(10);
    for n in (0..patterns.len()).step_by(30) {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            let matcher = get_matcher(patterns);
            b.iter(|| criterion::black_box(matcher.find_matches(graph)));
        });
    }
}

fn bench_trie_construction<M: Matcher, F: FnMut(Vec<Pattern>) -> M>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern],
    mut get_matcher: F,
) {
    group.sample_size(10);
    for n in (0..patterns.len()).step_by(30) {
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
    let patterns = glob::glob("datasets/small_circuits/*.bin")
        .expect("cannot read small circuits directory")
        .map(|p| {
            let g = rmp_serde::from_read(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small circuit"),
            )
            .expect("could not serialize");
            Pattern::from_graph(g).expect("pattern not connected")
        })
        .collect_vec();
    // TODO: use more than one graph in benchmark
    let graph = glob::glob("datasets/large_circuits/*.bin")
        .expect("cannot read large circuits directory")
        .map(|p| {
            rmp_serde::from_read(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small circuit"),
            )
            .expect("could not serialize")
        })
        .next()
        .expect("Did not find any large circuit");

    let mut group = c.benchmark_group("Many Patterns Matching");
    bench_matching(
        "Balanced Graph Trie",
        &mut group,
        &patterns,
        &graph,
        BalancedTrieMatcher::<BaseGraphTrie>::from_patterns,
    );
    bench_matching(
        "Balanced Graph Trie (cached)",
        &mut group,
        &patterns,
        &graph,
        |p| BalancedTrieMatcher::<BaseGraphTrie>::from_patterns(p).to_cached_trie(),
    );
    bench_matching(
        "Non-deterministic Graph Trie",
        &mut group,
        &patterns,
        &graph,
        NonDetTrieMatcher::<BaseGraphTrie>::from_patterns,
    );
    bench_matching(
        "Deterministic Graph Trie",
        &mut group,
        &patterns,
        &graph,
        DetTrieMatcher::<BaseGraphTrie>::from_patterns,
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
        BalancedTrieMatcher::<BaseGraphTrie>::from_patterns,
    );
    bench_trie_construction("Balanced Graph Trie (cached)", &mut group, &patterns, |p| {
        BalancedTrieMatcher::<BaseGraphTrie>::from_patterns(p).to_cached_trie()
    });
    bench_trie_construction(
        "Non-deterministic Graph Trie",
        &mut group,
        &patterns,
        NonDetTrieMatcher::<BaseGraphTrie>::from_patterns,
    );
    bench_trie_construction(
        "Deterministic Graph Trie",
        &mut group,
        &patterns,
        DetTrieMatcher::<BaseGraphTrie>::from_patterns,
    );
    group.finish();
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
