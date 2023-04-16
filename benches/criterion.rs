use std::fs::File;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use itertools::Itertools;

use portgraph::PortGraph;
use portmatching::matcher::many_patterns::LineGraphTrie;
use portmatching::matcher::many_patterns::ManyPatternMatcher;
use portmatching::matcher::many_patterns::NaiveManyMatcher;
use portmatching::pattern::Pattern;

fn bench<T: ManyPatternMatcher>(
    name: &str,
    c: &mut Criterion,
    patterns: &[Pattern],
    graph: &PortGraph,
) {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);
    for n in (0..patterns.len()).step_by(10) {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            let matcher: T = ManyPatternMatcher::from_patterns(patterns);
            b.iter(|| matcher.find_matches(&graph));
        });
    }
    group.finish();
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

    bench::<LineGraphTrie>("line-based graph trie", c, &patterns, &graph);
    bench::<NaiveManyMatcher>("naive", c, &patterns, &graph);
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
