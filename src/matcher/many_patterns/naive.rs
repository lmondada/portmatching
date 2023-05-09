#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    matcher::{Matcher, SinglePatternMatcher},
    pattern::Pattern,
};

use super::{ManyPatternMatcher, PatternID, PatternMatch};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
#[derive(Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NaiveManyMatcher {
    matchers: Vec<SinglePatternMatcher>,
}

impl Matcher for NaiveManyMatcher {
    type Match = PatternMatch;

    fn find_anchored_matches(
        &self,
        graph: &portgraph::PortGraph,
        root: portgraph::NodeIndex,
    ) -> Vec<Self::Match> {
        self.matchers
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.find_anchored_matches(graph, root)
                    .into_iter()
                    .map(move |m| PatternMatch {
                        id: PatternID(i),
                        root: m[&root],
                    })
            })
            .collect()
    }
}

impl ManyPatternMatcher for NaiveManyMatcher {
    fn add_pattern(&mut self, pattern: Pattern) -> PatternID {
        self.matchers
            .push(SinglePatternMatcher::from_pattern(pattern));
        PatternID(self.matchers.len() - 1)
    }
}
