use std::{
    fs, io,
    path::{Path, PathBuf},
};

use itertools::Itertools;
// use portgraph::dot::dot_string;
use portgraph::PortGraph;
use portmatching::{
    matcher::{
        many_patterns::{NaiveManyPatternMatcher, PatternID, PatternMatch},
        Matcher,
    },
    pattern::Pattern,
};

fn valid_binary_file(s: &str, pattern: &str) -> bool {
    s.starts_with(pattern) && s.ends_with(".bin")
}

fn load_patterns(dir: &Path) -> io::Result<Vec<Pattern>> {
    let mut patterns = Vec::new();
    let mut all_patterns: Vec<_> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let Ok(entry) = entry else { return None };
            let file_name = entry.file_name().to_str().unwrap().to_string();
            let path = entry.path();
            valid_binary_file(&file_name, "pattern").then_some(path)
        })
        .collect();
    all_patterns.sort_unstable();
    for path in all_patterns {
        let p: PortGraph = rmp_serde::from_read(fs::File::open(&path)?).unwrap();
        // {
        //     let mut path = path;
        //     path.set_extension("gv");
        //     fs::write(path, dot_string(&p)).unwrap();
        // }
        patterns.push(Pattern::from_graph(p).unwrap());
    }

    Ok(patterns)
}

fn load_graph(dir: &Path) -> io::Result<PortGraph> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_binary_file(&file_name, "graph") {
            let graph: PortGraph = rmp_serde::from_read(fs::File::open(&path)?).unwrap();
            // {
            //     let mut path = path;
            //     path.set_extension("gv");
            //     fs::write(path, dot_string(&graph)).unwrap();
            // }
            return Ok(graph);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn load_results(dir: &Path) -> io::Result<Vec<Vec<PatternMatch>>> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_binary_file(&file_name, "results") {
            let graph: Vec<Vec<PatternMatch>> =
                rmp_serde::from_read(fs::File::open(&path)?).unwrap();
            return Ok(graph);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

#[test]
fn from_saved_patterns() {
    let testcases = [
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelveth",
        "thirteenth",
        "fourteenth",
        "fifteenth",
        "sixteenth",
        "seventeenth",
        "eighteenth",
    ];
    for test in testcases {
        // println!("{test}...");
        let path: PathBuf = ["tests", "saved_patterns", test].iter().collect();
        let patterns = load_patterns(&path).unwrap();
        let graph = load_graph(&path).unwrap();
        let exp = load_results(&path).unwrap();

        let matcher = NaiveManyPatternMatcher::from_patterns(patterns.clone());
        // {
        //     let mut path = path;
        //     path.push("patterntrie.gv");
        //     fs::write(path, matcher.dotstring()).unwrap();
        // }
        let many_matches = matcher.find_matches(&graph);
        let many_matches = (0..patterns.len())
            .map(|i| {
                many_matches
                    .iter()
                    .filter(|m| m.id == PatternID(i))
                    .cloned()
                    .collect_vec()
            })
            .collect_vec();
        assert_eq!(many_matches, exp);
    }
}