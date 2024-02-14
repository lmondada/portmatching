use crate::{matcher::SinglePatternMatcher, Pattern};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
pub struct NaiveManyMatcher<P: Pattern> {
    matchers: Vec<SinglePatternMatcher<P>>,
}

impl<P: Pattern> NaiveManyMatcher<P> {
    pub fn from_patterns(patterns: Vec<P>) -> Self {
        Self {
            matchers: patterns
                .into_iter()
                .map(SinglePatternMatcher::new)
                .collect(),
        }
    }
}

impl<P: Pattern> Default for NaiveManyMatcher<P> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

// impl<U, G> PortMatcher<G, NodeIndex, U> for NaiveManyMatcher<U, (), UnweightedEdge>
// where
//     U: Universe,
//     G: LinkView + Copy,
// {
//     type PNode = ();
//     type PEdge = UnweightedEdge;

//     fn find_rooted_matches(&self, graph: G, root: NodeIndex) -> Vec<Match> {
//         self.matchers
//             .iter()
//             .enumerate()
//             .flat_map(|(i, m)| {
//                 m.find_rooted_matches(graph, root).into_iter().map(
//                     move |PatternMatch { root, .. }| PatternMatch {
//                         pattern: i.into(),
//                         root,
//                     },
//                 )
//             })
//             .collect()
//     }

//     fn get_pattern(&self, id: PatternID) -> Option<&Pattern<U, (), UnweightedEdge>> {
//         let m = self.matchers.get(id.0)?;
//         <SinglePatternMatcher<_, _, _> as PortMatcher<G, NodeIndex, U>>::get_pattern(m, 0.into())
//     }
// }

impl<P: Pattern> FromIterator<SinglePatternMatcher<P>> for NaiveManyMatcher<P> {
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<P>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod naive_tests {}
