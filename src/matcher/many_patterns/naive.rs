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

type Graph<'g> = (&'g PortGraph, NodeIndex);

impl<'g, C> Matcher<Graph<'g>> for NaiveManyMatcher<C> {
    type Match = PatternMatch;

    fn find_anchored_matches(&self, graph @ (_, root): Graph<'g>) -> Vec<Self::Match> {
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

impl<'g, C, P> ManyPatternMatcher<Graph<'g>, P> for NaiveManyMatcher<C>
where
    P: Pattern<Constraint = C> + 'static,
{
    type Constraint = C;

    fn add_pattern(&mut self, pattern: P) -> PatternID {
        self.matchers
            .push(SinglePatternMatcher::from_pattern(Box::new(pattern)));
        PatternID(self.matchers.len() - 1)
    }
}
