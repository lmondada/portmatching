use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use petgraph::visit::{GraphBase, IntoNodeIdentifiers};
use portgraph::{dot::DotFormat, LinkView, MultiPortGraph, NodeIndex, PortGraph};
use portmatching::{
    matcher::{ManyMatcher, PatternMatch, PortMatcher},
    PatternID, UnweightedPattern,
};

const DBG_DUMP_FILES: bool = false;

fn valid_json_file(s: &str, pattern: &str) -> bool {
    s.starts_with(pattern) && s.ends_with(".json")
}

fn load_patterns<G>(dir: &Path) -> io::Result<Vec<UnweightedPattern>>
where
    G: for<'de> serde::Deserialize<'de> + LinkView,
    for<'g> &'g G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
{
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
        let (p, root): (G, NodeIndex) = serde_json::from_reader(fs::File::open(&path)?).unwrap();
        if DBG_DUMP_FILES {
            let mut path = path;
            path.set_extension("gv");
            fs::write(path, p.dot_string()).unwrap();
        }
        patterns.push(UnweightedPattern::from_rooted_portgraph(&p, root));
    }

    Ok(patterns)
}

fn load_graph<G>(dir: &Path) -> io::Result<G>
where
    G: for<'de> serde::Deserialize<'de>,
    G: LinkView,
{
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "graph") {
            let graph: G = serde_json::from_reader(fs::File::open(&path)?).unwrap();
            if DBG_DUMP_FILES {
                let mut path = path;
                path.set_extension("kv");
                fs::write(path, graph.dot_string()).unwrap();
            }
            return Ok(graph);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn load_results(dir: &Path) -> io::Result<BTreeSet<PatternMatch<PatternID, NodeIndex>>> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "results") {
            let res: BTreeSet<PatternMatch<PatternID, NodeIndex>> =
                serde_json::from_reader(fs::File::open(path)?).unwrap();
            return Ok(res);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn test<G>(test_path: &Path)
where
    G: for<'de> serde::Deserialize<'de> + LinkView,
    for<'g> &'g G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
{
    let patterns = load_patterns::<G>(test_path).unwrap();
    let graph = load_graph::<G>(test_path).unwrap();
    let exp = load_results(test_path).unwrap();

    let matcher = ManyMatcher::from_patterns(patterns.clone());
    if DBG_DUMP_FILES {
        let mut path = test_path.to_owned();
        path.push("trie.gv");
        fs::write(path, matcher.dot_string()).unwrap();
    }
    let many_matches: BTreeSet<_> = matcher.find_matches(&graph).into_iter().collect();
    dbg!(many_matches.len(), exp.len());
    assert_eq!(many_matches, exp);
}

#[test]
fn saved_testcases() {
    let portgraph_testcases = ["0", "1", "2", "3"];
    let multiportgraph_testcases = ["0", "1"];
    for test_name in portgraph_testcases {
        let path: PathBuf = ["tests", "testcases", "portgraph", test_name]
            .iter()
            .collect();
        println!("{path:?}...");
        test::<PortGraph>(&path);
    }
    for test_name in multiportgraph_testcases {
        let path: PathBuf = ["tests", "testcases", "multiportgraph", test_name]
            .iter()
            .collect();
        println!("{path:?}...");
        test::<MultiPortGraph>(&path);
    }
}
