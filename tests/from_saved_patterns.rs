use std::{
    fs, io,
    path::{Path, PathBuf},
};

use itertools::Itertools;
use portgraph::dot::dot_string;
use portgraph::{NodeIndex, PortGraph};
use portmatching::{
    matcher::{
        many_patterns::{ManyPatternMatcher, PatternID, PatternMatch, TrieMatcher},
        Matcher,
    },
    pattern::UnweightedPattern,
};

fn valid_json_file(s: &str, pattern: &str) -> bool {
    s.starts_with(pattern) && s.ends_with(".json")
}

fn load_patterns(dir: &Path) -> io::Result<Vec<UnweightedPattern>> {
    let mut patterns = Vec::new();
    let mut all_patterns: Vec<_> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let Ok(entry) = entry else { return None };
            let file_name = entry.file_name().to_str().unwrap().to_string();
            let path = entry.path();
            valid_json_file(&file_name, "pattern").then_some(path)
        })
        .collect();
    all_patterns.sort_unstable();
    for path in all_patterns {
        let p: PortGraph = serde_json::from_reader(fs::File::open(&path)?).unwrap();
        {
            let mut path = path;
            path.set_extension("gv");
            fs::write(path, dot_string(&p)).unwrap();
        }
        patterns.push(UnweightedPattern::from_graph(p).unwrap());
    }

    Ok(patterns)
}

fn load_graph(dir: &Path) -> io::Result<PortGraph> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "graph") {
            let graph: PortGraph = serde_json::from_reader(fs::File::open(&path)?).unwrap();
            {
                let mut path = path;
                path.set_extension("gv");
                fs::write(path, dot_string(&graph)).unwrap();
            }
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
        if valid_json_file(&file_name, "results") {
            let res: Vec<Vec<PatternMatch>> =
                serde_json::from_reader(fs::File::open(&path)?).unwrap();
            return Ok(res);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn test<'g, M: Matcher<(&'g PortGraph, NodeIndex), Match = PatternMatch>>(
    matcher: &M,
    graph: &'g PortGraph,
    exp: &[Vec<PatternMatch>],
    n_patterns: usize,
) {
    let many_matches = matcher.find_matches(&graph);
    let many_matches = (0..n_patterns)
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
        "ninteenth",
        "twentieth",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
    ];
    for test_name in testcases {
        println!("{test_name}...");
        let path: PathBuf = ["tests", "saved_patterns", test_name].iter().collect();
        let patterns = load_patterns(&path).unwrap();
        let graph = load_graph(&path).unwrap();
        let exp = load_results(&path).unwrap();

        let mut matcher = TrieMatcher::from_patterns(patterns.clone());
        // {
        //     let mut path = path.clone();
        //     path.push("trie.gv");
        //     fs::write(path, matcher.dotstring()).unwrap();
        // }
        test(&matcher, &graph, &exp, patterns.len());
        matcher.optimise(1);
        test(&matcher, &graph, &exp, patterns.len());
    }
}
