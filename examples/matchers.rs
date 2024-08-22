use std::fs::File;
use std::io::Write;
use std::{borrow::Borrow, fmt::Debug};

use itertools::Itertools;
use portgraph::{render::DotFormat, LinkMut, PortGraph, PortMut};
use portmatching::matrix::MatrixPatternPosition;
use portmatching::string::StringPatternPosition;
use portmatching::{
    matrix::{MatrixManyMatcher, MatrixPattern, MatrixString},
    portgraph::{indexing::PGIndexKey, PGManyPatternMatcher, PGPattern},
    string::{StringManyMatcher, StringPattern},
    IndexMap, ManyMatcher, PatternMatch, PortMatcher,
};

fn main() {
    string_matching();
    matrix_matching();
    portgraph_matching();
}

fn string_matching() {
    let patterns = ["he$a$bo", "hello, world"];
    let patterns = patterns.map(StringPattern::parse_str).to_vec();

    let matcher = StringManyMatcher::from_patterns_with_det_heuristic(patterns, |_| true);

    let text = "hey Enschede, hello world".to_string();
    let matches = matcher.find_matches(&text).collect_vec();
    print_matches(&matches, &StringPatternPosition::start());

    // Write the matcher's dotstring to a file
    save_dot_string(&matcher, "target/out.dot");
}

fn matrix_matching() {
    let patterns = ["hello\n--rld", "hello\nwo---", "h----\n-e---\n--llo"];
    let patterns = patterns.map(MatrixPattern::parse_str).to_vec();
    dbg!(&patterns);

    let matcher = MatrixManyMatcher::from_patterns_with_det_heuristic(patterns, |_| true);

    let text = MatrixString::from(
        r#"
Enschede, hello
----------world
"#,
    );
    let matches = matcher.find_matches(&text).collect_vec();
    print_matches(&matches, &MatrixPatternPosition::start());

    // Write the matcher's dotstring to a file
    save_dot_string(&matcher, "target/out.dot");
}

fn portgraph_matching() {
    let p1 = {
        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 2);
        let n1 = g.add_node(1, 2);
        let n2 = g.add_node(1, 2);
        let n3 = g.add_node(2, 3);

        g.link_nodes(n0, 0, n1, 0).unwrap();
        g.link_nodes(n0, 1, n2, 0).unwrap();
        g.link_nodes(n1, 1, n3, 0).unwrap();
        g.link_nodes(n2, 1, n3, 1).unwrap();

        println!("Pattern 1\n{}", g.dot_string());

        g
    };
    let p2 = {
        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 2);
        let n1 = g.add_node(1, 1);
        let n2 = g.add_node(2, 0);

        g.link_nodes(n0, 0, n1, 0).unwrap();
        g.link_nodes(n0, 1, n2, 0).unwrap();
        g.link_nodes(n1, 0, n2, 1).unwrap();

        println!("Pattern 2\n{}", g.dot_string());

        g
    };

    let subject = p1.clone();

    let patterns = [p1, p2];
    let patterns = patterns.map(PGPattern::from_host_pick_root).to_vec();

    let matcher = PGManyPatternMatcher::from_patterns_with_det_heuristic(patterns, |_| false);

    let matches = matcher.find_matches(&subject).collect_vec();

    print_matches(&matches, &PGIndexKey::root(0));

    // Write the matcher's dotstring to a file
    save_dot_string(&matcher, "target/out.dot");
}

fn save_dot_string<PT, K: Debug, P: Debug, I>(matcher: &ManyMatcher<PT, K, P, I>, filename: &str) {
    let dot_string = matcher.dot_string();
    let mut file = File::create(filename).expect("Failed to create file");
    file.write_all(dot_string.as_bytes())
        .expect("Failed to write to file");
    println!("Matcher's dotstring written to '{}'", filename);
}

fn print_matches<K, V: Debug, M: IndexMap<Key = K, Value = V>>(
    matches: &[PatternMatch<M>],
    root_pos: &K,
) {
    for m in matches {
        println!(
            "Pattern {} matches at {:?}",
            m.pattern.0,
            m.match_data.get(root_pos).unwrap().borrow()
        );
    }
}
