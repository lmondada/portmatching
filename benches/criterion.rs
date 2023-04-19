use std::fs::File;

use criterion::BenchmarkGroup;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::measurement::WallTime;
use itertools::Itertools;

use portgraph::PortGraph;
use portmatching::matcher::many_patterns::LineGraphTrie;
use portmatching::matcher::many_patterns::ManyPatternMatcher;
use portmatching::matcher::many_patterns::NaiveManyMatcher;
use portmatching::matcher::many_patterns::graph_tries::BaseGraphTrie;
use portmatching::pattern::Pattern;

fn bench<T: ManyPatternMatcher>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern],
    graph: &PortGraph,
) {
    group.sample_size(10);
    for n in (0..patterns.len()).step_by(10) {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            let matcher: T = ManyPatternMatcher::from_patterns(patterns);
            b.iter(|| matcher.find_matches(&graph));
        });
    }
}

fn perform_benches(c: &mut Criterion) {
    let patterns = glob::glob("datasets/small_graphs/*.bin")
        .expect("cannot read small graphs directory")
        .map(|p| {
            let g = rmp_serde::from_read(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small graph"),
            )
            .expect("could not serialize");
            Pattern::from_graph(g).expect("pattern not connected")
        })
        .collect_vec();
    let graph = glob::glob("datasets/large_graphs/*.bin")
        .expect("cannot read large graphs directory")
        .map(|p| {
            rmp_serde::from_read(
                File::open(p.as_ref().expect("path does not exist?"))
                    .expect("Could not open small graph"),
            )
            .expect("could not serialize")
        })
        .next()
        .expect("Did not find any large graph");

    let mut group = c.benchmark_group("Many Patterns Matching");
    bench::<LineGraphTrie<BaseGraphTrie>>("Line-based Graph Trie", &mut group, &patterns, &graph);
    bench::<NaiveManyMatcher>("Naive", &mut group, &patterns, &graph);
    group.finish();
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
