use std::time::Instant;

use itertools::Itertools;
use portmatching::{
    concrete::matrix::{MatrixManyMatcher, MatrixPattern, MatrixString},
    PortMatcher,
};

struct Metrics {
    build_time: std::time::Duration,
    match_time: std::time::Duration,
    n_states: usize,
}

impl Metrics {
    fn report(&self) {
        println!("Build time:       {}ms", self.build_time.as_millis());
        println!("Match time:       {}ms", self.match_time.as_millis());
        println!("Number of states: {}", self.n_states);
    }
}

fn main() {
    let subject = MatrixString::from("");

    let patterns = vec![
        "--a$ca\n-\n$c\n",
        "\n---\n-\na\n",
        "\na---a\n",
        "-\n---a\n\n",
        "---a-aaaa\n",
        "\n-a-\na\n",
        "\n--a\n-aa\n\n\n",
    ]
    .into_iter()
    .map(MatrixPattern::parse_str)
    .collect_vec();

    let build_start = Instant::now();
    let matcher = MatrixManyMatcher::try_from_patterns(patterns, Default::default()).unwrap();
    let build_time = build_start.elapsed();

    let match_start = Instant::now();
    let _ = matcher.find_matches(&subject);
    let match_time = match_start.elapsed();

    let metrics = Metrics {
        build_time,
        match_time,
        n_states: matcher.n_states(),
    };
    metrics.report();
}
