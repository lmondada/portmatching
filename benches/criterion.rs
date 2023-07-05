use std::fs;
use std::fs::File;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use itertools::Itertools;

use portgraph::NodeIndex;
use portgraph::PortGraph;
use portgraph::UnmanagedDenseMap;
use portmatching::matcher::many_patterns::ManyMatcher;
use portmatching::matcher::PortMatcher;
use portmatching::matcher::UnweightedManyMatcher;
use portmatching::NaiveManyMatcher;
use portmatching::Pattern;
use portmatching::Property;
use rand::Rng;

fn bench_matching<'g, M: PortMatcher<&'g PortGraph>>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern<NodeIndex, M::PNode, M::PEdge>],
    sizes: impl Iterator<Item = usize>,
    graph: &'g PortGraph,
    mut get_matcher: impl FnMut(Vec<Pattern<NodeIndex, M::PNode, M::PEdge>>) -> M,
) where
    M::PEdge: Property,
    M::PNode: Property,
    NodeIndex: Copy,
{
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

fn bench_matching_xxl(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    prefix: &str,
    sizes: impl Iterator<Item = usize>,
    graph: &PortGraph,
) {
    group.sample_size(10);
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        let file_name = format!("datasets/xxl/tries/{prefix}_{size}.bin");
        let matcher: UnweightedManyMatcher =
            rmp_serde::from_read(fs::File::open(file_name).unwrap())
                .expect("could not deserialize trie");
        group.bench_with_input(BenchmarkId::new(name, size), &size, |b, _| {
            b.iter(|| criterion::black_box(matcher.find_matches(graph)));
        });
    }
}

#[allow(unused)]
fn gen_weights(nodes: impl Iterator<Item = NodeIndex>) -> UnmanagedDenseMap<NodeIndex, usize> {
    let mut rng = rand::thread_rng();
    let mut weights = UnmanagedDenseMap::new();
    for n in nodes {
        weights[n] = rng.gen_range(0..8);
    }
    weights
}

#[allow(unused)]
fn bench_matching_xxl_weighted(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    prefix: &str,
    sizes: impl Iterator<Item = usize>,
    graph: &PortGraph,
) {
    todo!()
    // let weights = gen_weights(graph.nodes_iter());
    // group.sample_size(10);
    // for size in sizes {
    //     group.throughput(Throughput::Elements(size as u64));
    //     let file_name = format!("datasets/xxl/tries/weighted_{prefix}_{size}.bin");
    //     let matcher: TrieMatcher<WeightedAdjConstraint<usize>, Address, WeightedPattern<usize>> =
    //         rmp_serde::from_read(fs::File::open(file_name).unwrap())
    //             .expect("could not deserialize trie");
    //     group.bench_with_input(BenchmarkId::new(name, size), &size, |b, _| {
    //         b.iter(|| criterion::black_box(matcher.find_weighted_matches(graph, &weights)));
    //     });
    // }
}

fn bench_trie_construction<'g, M: PortMatcher<&'g PortGraph>>(
    name: &str,
    group: &mut BenchmarkGroup<WallTime>,
    patterns: &[Pattern<NodeIndex, M::PNode, M::PEdge>],
    sizes: impl Iterator<Item = usize>,
    mut get_matcher: impl FnMut(Vec<Pattern<NodeIndex, M::PNode, M::PEdge>>) -> M,
) where
    M::PNode: Copy,
    M::PEdge: Copy,
{
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
            Pattern::from_portgraph(&g)
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
        ManyMatcher::from_patterns,
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
        ManyMatcher::from_patterns,
    );
    group.finish();

    let mut group = c.benchmark_group("Many Patterns Matching XXL");
    bench_matching_xxl(
        "Balanced Graph Trie",
        &mut group,
        "balanced",
        (500..=5000).step_by(500),
        &graph,
    );
    group.finish();

    // let mut group = c.benchmark_group("Many Patterns Matching XXL weighted");
    // bench_matching_xxl_weighted(
    //     "Balanced Graph Trie",
    //     &mut group,
    //     "balanced",
    //     (500..=5000).step_by(500),
    //     &graph,
    // );
    // group.finish();
}

criterion_group!(benches, perform_benches);
criterion_main!(benches);
