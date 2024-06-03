use crate::{
    constraint::Constraint,
    matcher::{Match, PatternMatch, PortMatcher, SinglePatternMatcher},
    pattern::Pattern,
    predicate::{AssignPredicate, FilterPredicate},
    Universe, VariableNaming,
};

/// A simple matcher for matching multiple patterns.
///
/// This matcher uses [`SinglePatternMatcher`]s to match each pattern separately.
/// Useful as a baseline in benchmarking.
pub struct NaiveManyMatcher<C, V, U> {
    matchers: Vec<SinglePatternMatcher<C, V, U>>,
}

impl<V, U, AP, FP> NaiveManyMatcher<Constraint<V, U, AP, FP>, V, U>
where
    V: VariableNaming,
    U: Universe,
{
    pub fn from_patterns<P>(patterns: Vec<P>) -> Self
    where
        AP: AssignPredicate<U, P::Host>,
        FP: FilterPredicate<U, P::Host>,
        P: Pattern<Constraint = Constraint<V, U, AP, FP>, U = U>,
    {
        Self {
            matchers: patterns
                .iter()
                .map(|p| SinglePatternMatcher::from_pattern(p))
                .collect(),
        }
    }
}

impl<C, V, U> Default for NaiveManyMatcher<C, V, U> {
    fn default() -> Self {
        Self {
            matchers: Default::default(),
        }
    }
}

impl<U, V, AP, FP, D> PortMatcher<U, D> for NaiveManyMatcher<Constraint<V, U, AP, FP>, V, U>
where
    V: VariableNaming,
    U: Universe,
    AP: AssignPredicate<U, D>,
    FP: FilterPredicate<U, D>,
{
    fn find_rooted_matches(&self, root_binding: U, host: &D) -> Vec<Match<U>> {
        self.matchers
            .iter()
            .enumerate()
            .flat_map(|(i, m)| {
                m.find_rooted_matches(root_binding.clone(), host)
                    .into_iter()
                    .map(move |PatternMatch { root, .. }| PatternMatch {
                        pattern: i.into(),
                        root,
                    })
            })
            .collect()
    }
}

impl<C, V, U> FromIterator<SinglePatternMatcher<C, V, U>> for NaiveManyMatcher<C, V, U> {
    fn from_iter<T: IntoIterator<Item = SinglePatternMatcher<C, V, U>>>(iter: T) -> Self {
        Self {
            matchers: iter.into_iter().collect(),
        }
    }
}
