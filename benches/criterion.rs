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
use portmatching::matcher::many_patterns::graph_tries::BaseGraphTrie;
use portmatching::matcher::many_patterns::LineGraphTrie;
use portmatching::matcher::many_patterns::ManyPatternMatcher;
use portmatching::matcher::many_patterns::NaiveManyMatcher;
use portmatching::matcher::Matcher;
use portmatching::pattern::Pattern;

fn bench<'graph, M: Matcher<'graph>, F: FnMut(Vec<Pattern>) -> M>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern],
    graph: &'graph PortGraph,
    mut get_matcher: F,
) {
    group.sample_size(10);
    for n in (0..patterns.len()).step_by(100) {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, &n| {
            let patterns = Vec::from_iter(patterns[0..n].iter().cloned());
            let matcher = get_matcher(patterns);
            b.iter(|| matcher.find_matches(graph));
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
    bench(
        "Line-based Graph Trie",
        &mut group,
        &patterns,
        &graph,
        LineGraphTrie::<BaseGraphTrie>::from_patterns,
    );
    bench("No Cached Graph Trie", &mut group, &patterns, &graph, |p| {
        LineGraphTrie::<BaseGraphTrie>::from_patterns(p).to_no_cached_trie()
    });
    bench("Cached Graph Trie", &mut group, &patterns, &graph, |p| {
        LineGraphTrie::<BaseGraphTrie>::from_patterns(p).to_cached_trie()
    });
    bench("Naive", &mut group, &patterns, &graph, |p| {
        NaiveManyMatcher::from_patterns(p)
    });
    group.finish();
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
