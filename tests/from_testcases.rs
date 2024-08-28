use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

use itertools::Itertools;
use portgraph::{render::DotFormat, PortGraph};
use portmatching::{
    matcher::PortMatcher,
    pattern::ConcretePattern,
    portgraph::{PGManyPatternMatcher, PGPattern},
    utils::test::SerialPatternMatch,
};

const DBG_DUMP_FILES: bool = true;

fn valid_json_file(s: &str, pattern: &str) -> bool {
    s.starts_with(pattern) && s.ends_with(".json")
}

fn load_patterns(dir: &Path) -> io::Result<Vec<PGPattern<PortGraph>>>
// where
//     G: for<'de> serde::Deserialize<'de> + LinkView,
//     for<'g> &'g G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
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
        let p: PGPattern<_> = serde_json::from_reader(fs::File::open(&path)?).unwrap();
        if DBG_DUMP_FILES {
            let mut path = path;
            path.set_extension("gv");
            fs::write(path, p.as_host().dot_string()).unwrap();
        }
        patterns.push(p);
    }

    Ok(patterns)
}

fn load_graph(dir: &Path) -> io::Result<PortGraph>
// where
//     G: for<'de> serde::Deserialize<'de>,
//     G: LinkView + DotFormat,
{
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "graph") {
            let graph: PortGraph = serde_json::from_reader(fs::File::open(&path)?).unwrap();
            if DBG_DUMP_FILES {
                let mut path = path;
                path.set_extension("gv");
                fs::write(path, graph.dot_string()).unwrap();
            }
            return Ok(graph);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn load_results(dir: &Path) -> io::Result<BTreeSet<SerialPatternMatch>> {
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let file_name = entry.file_name().to_str().unwrap().to_string();
        let path = entry.path();
        if valid_json_file(&file_name, "results") {
            let res: BTreeSet<SerialPatternMatch> =
                serde_json::from_reader(fs::File::open(path)?).unwrap();
            return Ok(res);
        }
    }

    Err(io::Error::new(io::ErrorKind::Other, "no file found"))
}

fn test(test_path: &Path)
// where
//     G: for<'de> serde::Deserialize<'de> + LinkView,
//     for<'g> &'g G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
{
    let patterns = load_patterns(test_path).unwrap();
    let graph = load_graph(test_path).unwrap();
    let exp = load_results(test_path).unwrap();

    // let mut cnt = 0;
    // let matcher = ManyMatcher::from_patterns_with_det_heuristic(patterns.clone(), |_| {
    //     cnt += 1;
    //     cnt <= 6
    // });
    let matcher = PGManyPatternMatcher::try_from_patterns(patterns.clone()).unwrap();
    if DBG_DUMP_FILES {
        let mut path = test_path.to_owned();
        path.push("trie.gv");
        fs::write(path, matcher.dot_string()).unwrap();
    }
    let many_matches: BTreeSet<SerialPatternMatch> =
        matcher.find_matches(&graph).map_into().collect();
    assert_eq!(many_matches, exp);
}

#[test]
fn saved_testcases() {
    let portgraph_testcases = ["0", "1", "2", "3", "4", "5"];
    // let multiportgraph_testcases = ["0", "1"];
    for test_name in portgraph_testcases {
        let path: PathBuf = ["tests", "testcases", "portgraph", test_name]
            .iter()
            .collect();
        println!("{path:?}...");
        test(&path);
    }
    // for test_name in multiportgraph_testcases {
    //     let path: PathBuf = ["tests", "testcases", "multiportgraph", test_name]
    //         .iter()
    //         .collect();
    //     println!("{path:?}...");
    //     test::<MultiPortGraph>(&path);
    // }
}
