use portgraph::{NodeIndex, PortGraph};

use crate::{
    matcher::{Matcher, SinglePatternMatcher},
    Pattern,
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

impl<C> Matcher for NaiveManyMatcher<C> {
    type Match = PatternMatch;
    type Graph<'g> = (&'g PortGraph, NodeIndex);

    fn find_anchored_matches<'g>(&self, graph @ (_, root): Self::Graph<'g>) -> Vec<Self::Match> {
        self.matchers
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.find_anchored_matches(graph)
                    .into_iter()
                    .map(move |m| PatternMatch {
                        id: PatternID(i),
                        root: m[&root],
                    })
            })
            .collect()
    }
}

impl<C> ManyPatternMatcher for NaiveManyMatcher<C> {
    type Constraint = C;

    fn add_pattern(&mut self, pattern: impl Pattern<Constraint = C> + 'static) -> PatternID {
        self.matchers
            .push(SinglePatternMatcher::from_pattern(Box::new(pattern)));
        PatternID(self.matchers.len() - 1)
    }
}
