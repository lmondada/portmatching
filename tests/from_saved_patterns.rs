use std::{
    fs, io,
    path::{Path, PathBuf},
};

use itertools::Itertools;

use portgraph::{NodeIndex, PortGraph, PortOffset};
use portmatching::{
    matcher::{ManyMatcher, PatternMatch, PortMatcher},
    PatternID, Universe, UnweightedPattern,
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
        // {
        //     let mut path = path;
        //     path.set_extension("gv");
        //     fs::write(path, dot_string(&p)).unwrap();
        // }
        patterns.push(UnweightedPattern::from_portgraph(&p));
    }

    Ok(patterns)
}

fn load_graph(dir: &Path) -> io::Result<PortGraph> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "graph") {
            let graph: PortGraph = serde_json::from_reader(fs::File::open(path)?).unwrap();
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

fn load_results(dir: &Path) -> io::Result<Vec<PatternMatch<PatternID, NodeIndex>>> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "results") {
            let res: Vec<PatternMatch<PatternID, NodeIndex>> =
                serde_json::from_reader(fs::File::open(path)?).unwrap();
            return Ok(res);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn test<'g, M, U>(
    matcher: &M,
    graph: &'g PortGraph,
    exp: &Vec<PatternMatch<PatternID, NodeIndex>>,
    n_patterns: usize,
) where
    M: PortMatcher<&'g PortGraph, U, PNode = (), PEdge = (PortOffset, PortOffset)>,
    U: Universe,
{
    let many_matches = matcher.find_matches(graph);
    assert_eq!(&many_matches, exp);
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
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
    ];
    for test_name in testcases {
        println!("{test_name}...");
        let path: PathBuf = ["tests", "saved_patterns", test_name].iter().collect();
        let patterns = load_patterns(&path).unwrap();
        let graph = load_graph(&path).unwrap();
        let exp = load_results(&path).unwrap();

        let mut matcher = ManyMatcher::from_patterns(patterns.clone());
        // {
        //     let mut path = path.clone();
        //     path.push("trie.gv");
        //     fs::write(path, matcher.dotstring()).unwrap();
        // }
        test(&matcher, &graph, &exp, patterns.len());
    }
}
