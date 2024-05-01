//! A simple matcher for a single pattern.
//!
//! This matcher is used as a baseline in benchmarking by repeating
//! the matching process for each pattern separately.
use std::{collections::VecDeque, fmt::Debug, hash::Hash};

use crate::{
    new_api_v04::{
        constraint::Constraint,
        predicate::{AssignPredicate, FilterPredicate},
    },
    pattern::Pattern,
    HashMap, Universe,
};

use super::{Match, PatternMatch, PortMatcher, VariableNaming};

/// A simple matcher for a single pattern.
pub struct SinglePatternMatcher<C, V, U> {
    /// The constraints forming the pattern
    constraints: Vec<C>,
    /// A map from variables to pattern nodes
    variable_to_pattern: HashMap<V, U>,
}

impl<U, V, AP, FP, D> PortMatcher<U, D> for SinglePatternMatcher<Constraint<V, U, AP, FP>, V, U>
where
    V: VariableNaming,
    U: Universe,
    AP: AssignPredicate<U, D>,
    FP: FilterPredicate<U, D>,
{
    fn find_rooted_matches(&self, root: U, graph: &D) -> Vec<Match<U>> {
        self.find_rooted_match(root, graph)
    }
}

impl<V, U, AP, FP> SinglePatternMatcher<Constraint<V, U, AP, FP>, V, U>
where
    V: VariableNaming,
    U: Clone + Debug,
{
    /// Create a matcher from a vector of constraints.
    ///
    /// The root variable name is obtained from the variable naming convention.
    pub fn from_pattern<P>(pattern: &P) -> Self
    where
        AP: AssignPredicate<U, P::Host>,
        FP: FilterPredicate<U, P::Host>,
        P: Pattern<Constraint = Constraint<V, U, AP, FP>, U = U>,
    {
        let constraints = pattern.to_constraint_vec();
        let pattern_root = pattern.root();
        let pattern_host = pattern.as_host();
        // Match pattern on itself to get map from variables to pattern nodes
        let scopes = get_scopes(constraints.as_slice(), pattern_root.clone(), pattern_host);
        if scopes.is_empty() {
            panic!("Pattern does not match itself");
        }
        let variable_to_pattern = scopes.into_iter().next().unwrap();
        Self {
            constraints,
            variable_to_pattern,
        }
    }
}

impl<U, V, AP, FP> SinglePatternMatcher<Constraint<V, U, AP, FP>, V, U>
where
    V: VariableNaming,
    U: Universe,
{
    /// Whether `self` matches `host` anchored at `root`.
    ///
    /// Check whether each edge of the pattern is valid in the host
    fn match_exists<D>(&self, root_binding: U, host: &D) -> bool
    where
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
        U: Hash,
    {
        self.get_match_map(root_binding, host).next().is_some()
    }

    /// Match the pattern and return a map from pattern nodes to host nodes
    ///
    /// Returns `None` if the pattern does not match.
    pub fn get_match_map<D>(
        &self,
        root_binding: U,
        host: &D,
    ) -> impl Iterator<Item = HashMap<U, U>> + '_
    where
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
        U: Universe,
    {
        let all_scopes = get_scopes(&self.constraints, root_binding, host);
        all_scopes.into_iter().map(|scope| {
            scope
                .into_iter()
                .map(|(v, u)| (self.variable_to_pattern[&v].clone(), u))
                .collect()
        })
    }

    /// The matches in `host` starting at `host_root`
    ///
    /// For single pattern matchers there is always at most one match
    fn find_rooted_match<D>(&self, root_binding: U, host: &D) -> Vec<Match<U>>
    where
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
        U: Hash,
    {
        if self.match_exists(root_binding.clone(), host) {
            vec![PatternMatch {
                pattern: 0.into(),
                root: root_binding.clone(),
            }]
        } else {
            Vec::new()
        }
    }
}

/// Get the valid scope assignments as a map from variable names to host nodes.
fn get_scopes<V, U, AP, FP, D>(
    constraints: &[Constraint<V, U, AP, FP>],
    root_binding: U,
    host: &D,
) -> Vec<HashMap<V, U>>
where
    AP: AssignPredicate<U, D>,
    FP: FilterPredicate<U, D>,
    U: Clone + Debug,
    V: VariableNaming,
{
    let mut candidates = VecDeque::new();
    candidates.push_back((
        constraints,
        HashMap::from_iter([(V::root_variable(), root_binding)]),
    ));
    let mut final_match_maps = Vec::new();
    while let Some((constraints, match_map)) = candidates.pop_front() {
        let [constraint, remaining @ ..] = constraints else {
            final_match_maps.push(match_map);
            continue;
        };

        candidates.extend(
            constraint
                .satisfy(host, match_map.clone())
                .unwrap()
                .into_iter()
                .map(|m| (remaining, m)),
        )
    }
    final_match_maps
}
