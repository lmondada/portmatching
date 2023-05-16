use crate::{
    matcher::{Matcher, SinglePatternMatcher},
    Constraint, Pattern,
};

use super::{ManyPatternMatcher, PatternID, PatternMatch};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
pub struct NaiveManyMatcher<C> {
    matchers: Vec<SinglePatternMatcher<Box<dyn Pattern<Constraint = C>>>>,
}

impl<C> Default for NaiveManyMatcher<C> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<C: Constraint> Matcher for NaiveManyMatcher<C> {
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

impl<C: Constraint> ManyPatternMatcher for NaiveManyMatcher<C> {
    type Constraint = C;

    fn add_pattern(&mut self, pattern: impl Pattern<Constraint = C> + 'static) -> PatternID {
        self.matchers
            .push(SinglePatternMatcher::from_pattern(Box::new(pattern)));
        PatternID(self.matchers.len() - 1)
    }
}
