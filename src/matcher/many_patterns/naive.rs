use std::borrow::Borrow;

use itertools::Itertools;
use portgraph::{NodeIndex, PortGraph, PortView};

use crate::{
    matcher::{Match, PortMatcher, SinglePatternMatcher},
    patterns::UnweightedEdge,
    Pattern, Property, Universe,
};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
pub struct NaiveManyMatcher<U: Universe, PNode, PEdge: Property> {
    matchers: Vec<SinglePatternMatcher<U, PNode, PEdge>>,
}

impl<U: Universe, PNode: Property, PEdge: Property> NaiveManyMatcher<U, PNode, PEdge> {
    pub fn from_patterns(patterns: Vec<Pattern<U, PNode, PEdge>>) -> Self {
        Self {
            matchers: patterns
                .into_iter()
                .map(SinglePatternMatcher::new)
                .collect(),
        }
    }
}

impl<U: Universe, PNode, PEdge: Property> Default for NaiveManyMatcher<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<G> PortMatcher<G> for NaiveManyMatcher<NodeIndex, (), UnweightedEdge>
where
    G: Borrow<PortGraph> + Copy,
{
    type PNode = ();
    type PEdge = UnweightedEdge;

    fn find_rooted_matches(&self, graph: G, root: NodeIndex) -> Vec<Match<'_, Self, G>> {
        self.matchers
            .iter()
            .flat_map(|m| m.find_rooted_matches(graph, root))
            .collect()
    }
}

impl<U: Universe, PNode, PEdge: Property> FromIterator<SinglePatternMatcher<U, PNode, PEdge>>
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
    use proptest::prelude::*;

    use crate::NaiveManyMatcher;
}
