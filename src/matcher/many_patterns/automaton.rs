use std::{
    fmt::{self, Debug},
    hash::Hash,
};

use itertools::Itertools;

use crate::{
    automaton::{AutomatonBuilder, ConstraintAutomaton},
    constraint::{Constraint, DetHeuristic},
    matcher::PatternMatch,
    mutex_tree::ToMutuallyExclusiveTree,
    pattern::Pattern,
    predicate::{ArityPredicate, AssignPredicate, FilterPredicate},
    HashMap, PatternID, SinglePatternMatcher, Universe, VariableNaming,
};

/// A graph pattern matcher using scope automata.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManyMatcher<P, C> {
    automaton: ConstraintAutomaton<C>,
    patterns: HashMap<PatternID, P>,
}

impl<P: Pattern<Constraint = Constraint<V, U, AP, FP>, Host = D, U = U>, V, U, AP, FP, D>
    ManyMatcher<P, Constraint<V, U, AP, FP>>
where
    AP: ArityPredicate,
    FP: ArityPredicate,
    Constraint<V, U, AP, FP>: Eq + Clone + ToMutuallyExclusiveTree,
{
    pub fn from_patterns(patterns: Vec<P>) -> Self
    where
        Constraint<V, U, AP, FP>: DetHeuristic,
    {
        Self::from_patterns_with_det_heuristic(patterns, Constraint::make_det)
    }

    pub fn from_patterns_with_det_heuristic(
        patterns: Vec<P>,
        make_det: impl for<'c> Fn(&[&'c Constraint<V, U, AP, FP>]) -> bool,
    ) -> Self
    where
        Constraint<V, U, AP, FP>: DetHeuristic,
    {
        let constraints = patterns.iter().map(|p| p.to_constraint_vec()).collect_vec();
        let builder = AutomatonBuilder::from_constraints(constraints);
        let (automaton, pattern_to_id) = builder.build(make_det);
        let patterns = pattern_to_id.into_iter().zip(patterns).collect();
        Self {
            automaton,
            patterns,
        }
    }
}

impl<P, C> fmt::Debug for ManyMatcher<P, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ManyMatcher {{ {} patterns }}", self.patterns.len())
    }
}

impl<P, V, U, AP, FP> ManyMatcher<P, Constraint<V, U, AP, FP>> {
    pub fn run<D>(&self, root_binding: U, host: &D) -> Vec<PatternMatch<PatternID, U>>
    where
        AP: AssignPredicate<U, D>,
        FP: FilterPredicate<U, D>,
        V: VariableNaming,
        U: Eq + Hash + Clone + Debug,
    {
        self.automaton
            .run(root_binding.clone(), host)
            .map(|id| PatternMatch::new(id, root_binding.clone()))
            .collect()
    }

    pub fn get_pattern(&self, id: PatternID) -> Option<&P> {
        self.patterns.get(&id)
    }
}

impl<P, V, U, AP, FP> ManyMatcher<P, Constraint<V, U, AP, FP>>
where
    P: Pattern<Constraint = Constraint<V, U, AP, FP>, U = U>,
    V: VariableNaming,
    U: Universe,
{
    pub fn get_match_map<D>(
        &self,
        m: PatternMatch<PatternID, U>,
        host: &P::Host,
    ) -> Vec<HashMap<U, U>>
    where
        AP: AssignPredicate<U, P::Host>,
        FP: FilterPredicate<U, P::Host>,
        P: Pattern<Constraint = Constraint<V, U, AP, FP>, U = U>,
    {
        let p = self.patterns.get(&m.pattern).unwrap();
        let single_matcher = SinglePatternMatcher::from_pattern(p);
        single_matcher
            .get_match_map(m.root, host)
            .map(|m| m.into_iter().collect())
            .collect()
    }
}

impl<C: Debug, P> ManyMatcher<P, C> {
    /// A dotstring representation of the trie.
    pub fn dot_string(&self) -> String {
        self.automaton.dot_string()
    }
}
