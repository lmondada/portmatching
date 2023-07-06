use std::{
    fs, io,
    path::{Path, PathBuf},
};

use portgraph::{dot::DotFormat, NodeIndex, PortGraph, PortOffset};
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
        let (p, root): (PortGraph, NodeIndex) =
            serde_json::from_reader(fs::File::open(&path)?).unwrap();
        {
            let mut path = path;
            path.set_extension("gv");
            fs::write(path, p.dot_string()).unwrap();
        }
        patterns.push(UnweightedPattern::from_rooted_portgraph(&p, root));
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
                fs::write(path, graph.dot_string()).unwrap();
            }
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

fn test<'g, M, U>(matcher: &M, graph: &'g PortGraph, exp: &Vec<PatternMatch<PatternID, NodeIndex>>)
where
    M: PortMatcher<&'g PortGraph, U, PNode = (), PEdge = (PortOffset, PortOffset)>,
    U: Universe,
{
    let many_matches = matcher.find_matches(graph);
    assert_eq!(&many_matches, exp);
}

#[test]
fn from_saved_patterns() {
    let testcases = ["0", "1"];
    for test_name in testcases {
        println!("{test_name}...");
        let path: PathBuf = ["tests", "saved_patterns", test_name].iter().collect();
        let patterns = load_patterns(&path).unwrap();
        let graph = load_graph(&path).unwrap();
        let exp = load_results(&path).unwrap();

        let matcher = ManyMatcher::from_patterns(patterns.clone());
        {
            let mut path = path.clone();
            path.push("trie.gv");
            fs::write(path, matcher.dot_string()).unwrap();
        }
        test(&matcher, &graph, &exp);
    }
}
