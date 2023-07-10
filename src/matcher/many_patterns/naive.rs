use std::{hash::Hash};


use portgraph::{NodeIndex, PortGraph};

use crate::{
    matcher::{Match, PatternMatch, PortMatcher, SinglePatternMatcher},
    patterns::UnweightedEdge,
    EdgeProperty, NodeProperty, Pattern, PatternID, Universe,
};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
pub struct NaiveManyMatcher<U: Universe, PNode, PEdge: Eq + Hash> {
    matchers: Vec<SinglePatternMatcher<U, PNode, PEdge>>,
}

impl<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty> NaiveManyMatcher<U, PNode, PEdge> {
    pub fn from_patterns(patterns: Vec<Pattern<U, PNode, PEdge>>) -> Self {
        Self {
            matchers: patterns
                .into_iter()
                .map(SinglePatternMatcher::new)
                .collect(),
        }
    }
}

impl<U: Universe, PNode, PEdge: Eq + Hash> Default for NaiveManyMatcher<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<U> PortMatcher<PortGraph, U> for NaiveManyMatcher<U, (), UnweightedEdge>
where
    U: Universe,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Match<PortGraph>> {
        self.matchers
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.find_rooted_matches(graph, root).into_iter().map(
                    move |PatternMatch { root, .. }| PatternMatch {
                        pattern: i.into(),
                        root,
                    },
                )
            })
            .collect()
    }

    fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, (), UnweightedEdge>> {
        let m = self.matchers.get(id.0)?;
        <SinglePatternMatcher<_, _, _> as PortMatcher<PortGraph, U>>::get_pattern(m, 0.into())
    }
}

impl<U: Universe, PNode, PEdge: Eq + Hash> FromIterator<SinglePatternMatcher<U, PNode, PEdge>>
    for NaiveManyMatcher<U, PNode, PEdge>
{
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<U, PNode, PEdge>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod naive_tests {
    

    
}
