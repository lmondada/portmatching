use portgraph::{PortGraph, NodeIndex};

pub mod single_pattern;
pub mod many_patterns;

pub use single_pattern::SinglePatternMatcher;

pub trait Matcher {
    type Match;

    fn find_anchored_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Self::Match>;

    fn find_matches(&self, graph: &PortGraph) -> Vec<Self::Match> {
        let mut matches = Vec::new();
        for root in graph.nodes_iter() {
            matches.append(&mut self.find_anchored_matches(graph, root));
        }
        matches
    }
}
