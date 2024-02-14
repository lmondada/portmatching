use std::{collections::VecDeque, hash::Hash, ops::Deref};

use delegate::delegate;
use itertools::Itertools;

use crate::{
    automaton::{ScopeAutomaton, StateID, Transition},
    constraint::ScopeConstraint,
    HashMap, Pattern, PatternID,
};

pub struct LineBuilder<C: ScopeConstraint> {
    /// The rest of the constraints to add
    patterns: Vec<Vec<C>>,
    /// The matcher being built
    matcher: ScopeAutomaton<C>,
    /// A cache of automaton state so that the same state is not inserted twice
    cached_states: HashMap<String, StateID>,
}

impl<C: ScopeConstraint + Clone> LineBuilder<C> {
    #[allow(unused)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_patterns<P>(patterns: Vec<P>) -> Self
    where
        P: Pattern<Constraint = C>,
    {
        patterns.iter().collect()
    }

    #[allow(unused)]
    pub fn add_pattern(&mut self, pattern: &impl Pattern<Constraint = C>) {
        self.patterns.push(pattern.constraints().collect());
    }

    pub fn build(mut self) -> ScopeAutomaton<C> {
        // Convert patterns to pattern in construction
        let patterns = self
            .patterns
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, p)| PatternInConstruction::new(p, PatternID(i)))
            .collect::<Vec<_>>();

        let mut to_insert = VecDeque::new();
        to_insert.push_back((self.matcher.root(), patterns));
        while let Some((state, patterns)) = to_insert.pop_front() {
            // Filter out done patterns
            let (done_patterns, patterns): (Vec<_>, Vec<_>) =
                patterns.into_iter().partition(|p| p.edges.is_empty());
            // Insert flags for done patterns
            for p in done_patterns {
                self.matcher.add_match(state, p.pattern_id);
            }
            let next_constraints = patterns.iter().map(|p| p.edges.front().unwrap());
            let constraints_split = C::split(next_constraints);
            let new_children = self.add_transitions(
                state,
                constraints_split.labels,
                constraints_split.deterministic,
            );
            // Enqueue the new states into queue
            for (new_state, constraint_set) in new_children
                .into_iter()
                .zip(constraints_split.constraint_sets)
            {
                let Some(new_state) = new_state else {
                    continue;
                };
                let patterns = constraint_set
                    .into_iter()
                    .map(|(i, consumed)| {
                        let mut p = patterns[i].clone();
                        if consumed {
                            p.pop_front();
                        }
                        p
                    })
                    .collect();
                to_insert.push_back((new_state, patterns));
            }
        }
        self.matcher
    }

    /// Add transitions to matcher
    ///
    /// Either deterministic or non-deterministic
    ///
    /// Returns the newly created states
    fn add_transitions(
        &mut self,
        state: StateID,
        constraints: Vec<Option<C>>,
        deterministic: bool,
    ) -> Vec<Option<StateID>> {
        if deterministic {
            for (i, c) in constraints.iter().enumerate() {
                let is_last = i == constraints.len() - 1;
                if c.is_none() && !is_last {
                    panic!("Only last constraint may be None for deterministc transition");
                }
            }
        } else {
            self.matcher.make_non_det(state);
        }
        let next_states = constraints
            .iter()
            .map(|c| {
                let id = c.as_ref()?.uid()?;
                self.cached_states.get(&id).copied()
            })
            .collect_vec();
        let scopes = constraints
            .iter()
            .map(|c| {
                let mut scope = self.matcher.scope(state).clone();
                if let Some(c) = c.as_ref() {
                    scope.extend(c.new_symbols());
                }
                scope
            })
            .collect_vec();
        let transitions: Vec<Transition<_>> = constraints.into_iter().map_into().collect();

        let new_states =
            self.matcher
                .set_children(state, transitions.iter().cloned(), &next_states, scopes);

        self.cache_new_states(new_states.iter().copied().zip(transitions));
        new_states
    }

    fn cache_new_states(
        &mut self,
        new_states: impl IntoIterator<Item = (Option<StateID>, Transition<C>)>,
    ) {
        for (new_state, transition) in new_states {
            let Some(new_state) = new_state else { continue };
            if let Transition::Constraint(c) = transition {
                if let Some(id) = c.uid() {
                    self.cached_states.entry(id).or_insert(new_state);
                }
            }
        }
    }
}

#[derive(Clone)]
struct PatternInConstruction<C> {
    /// Constraints of the pattern yet to be inserted
    edges: VecDeque<C>,
    /// Pattern ID
    pattern_id: PatternID,
    // /// The scope required to interpret the predicates
    // /// This is initially empty, but is updated as predicates are consumed
    // scope: HashSet<Symbol>,
}

// struct TransitionInConstruction<'a, U: Universe, PNode, PEdge: EdgeProperty> {
//     source: StateID,
//     target: Option<StateID>,
//     pred: EdgePredicate<PNode, PEdge, PEdge::OffsetID>,
//     patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
// }

// impl<'a, U: Universe, PNode, PEdge: EdgeProperty> TransitionInConstruction<'a, U, PNode, PEdge> {
//     fn scope(&self) -> HashSet<Symbol> {
//         self.patterns
//             .iter()
//             .flat_map(|p| p.scope.iter().copied())
//             .collect()
//     }
// }

impl<C> PatternInConstruction<C> {
    fn new(edges: impl IntoIterator<Item = C>, pattern_id: PatternID) -> Self {
        Self {
            edges: edges.into_iter().collect(),
            pattern_id,
        }
    }

    delegate! {
        to self.edges {
            fn pop_front(&mut self) -> Option<C>;
        }
    }
}

impl<C: ScopeConstraint + Clone> Default for LineBuilder<C> {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            matcher: ScopeAutomaton::new(),
            cached_states: HashMap::default(),
        }
    }
}

impl<'p, C: ScopeConstraint + Clone, P: Pattern<Constraint = C>> FromIterator<&'p P>
    for LineBuilder<C>
{
    fn from_iter<T: IntoIterator<Item = &'p P>>(iter: T) -> Self {
        Self {
            patterns: iter
                .into_iter()
                .map(|p| p.constraints().collect())
                .collect(),
            cached_states: HashMap::default(),
            matcher: ScopeAutomaton::new(),
        }
    }
}

fn partition_by<V, U, F>(iter: impl IntoIterator<Item = V>, f: F) -> HashMap<U, Vec<V>>
where
    U: Eq + Hash,
    F: for<'a> Fn(&'a mut V) -> U,
{
    let mut partitions = HashMap::default();
    for mut v in iter {
        let u = f(&mut v);
        partitions.entry(u).or_insert_with(Vec::new).push(v);
    }
    partitions
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::PortOffset;

    use crate::portgraph::pattern::{LinePattern, UnweightedEdge};
    use crate::portgraph::EdgeProperty;

    use super::LineBuilder;

    impl EdgeProperty for usize {
        type OffsetID = usize;

        fn reverse(&self) -> Option<Self> {
            Some(*self)
        }

        fn offset_id(&self) -> Self::OffsetID {
            0
        }
    }

    #[test]
    fn test_simple_build() {
        let mut p1 = LinePattern::new();
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, edge1), (1, 2, edge2)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1), (1, 2, edge2)]);

        let patterns = [p1, p2].into_iter().map(|lp| lp.to_pattern()).collect_vec();

        let builder = LineBuilder::from_patterns(patterns);
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 7);
    }

    #[test]
    #[should_panic]
    fn test_incompatible_preds() {
        let mut p1 = LinePattern::new();
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1), (0, 2, edge2)]);

        let builder: LineBuilder<_> = [p1.to_pattern(), p2.to_pattern()].iter().collect();
        builder.build();
    }

    #[test]
    fn test_build_two_dir() {
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);
        p1.add_line(0, vec![(0, 3, edge2), (3, 4, edge1)]);
        p1.add_line(1, vec![(1, 5, edge1)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);
        p2.add_line(0, vec![(0, 3, edge2), (3, 4, edge1)]);
        p2.add_line(3, vec![(3, 5, edge1)]);

        let builder: LineBuilder<_> = [p1.to_pattern(), p2.to_pattern()].iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 18);
    }

    #[test]
    fn test_build_two_different_length() {
        let mut p1 = LinePattern::new();
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);
        p1.add_line(0, vec![(0, 3, edge1), (3, 4, edge1)]);
        p1.add_line(1, vec![(1, 5, edge1)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1)]);
        p2.add_line(0, vec![(0, 3, edge1), (3, 4, edge1)]);

        let builder: LineBuilder<_> = [p1.to_pattern(), p2.to_pattern()].iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 15);
    }

    #[test]
    fn test_build_full_det() {
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(
            0,
            vec![(0, 1, edge1), (1, 2, edge1), (2, 3, edge1), (3, 4, edge1)],
        );
        p1.add_line(0, vec![(0, 1, edge2), (1, 5, edge1)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);
        p2.add_line(0, vec![(0, 1, edge2)]);

        let builder: LineBuilder<_> = [p1.to_pattern(), p2.to_pattern()].iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 13);
    }

    #[test]
    fn test_build_full_3_det() {
        let edge1 = (PortOffset::new_incoming(0), PortOffset::new_outgoing(0));
        let edge2 = (PortOffset::new_incoming(1), PortOffset::new_outgoing(1));
        let edge3 = (PortOffset::new_incoming(2), PortOffset::new_outgoing(2));
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(
            0,
            vec![(0, 1, edge1), (1, 2, edge1), (2, 3, edge1), (3, 4, edge1)],
        );
        p1.add_line(0, vec![(0, 1, edge2), (1, 5, edge1)]);
        p1.add_line(0, vec![(0, 1, edge3), (1, 6, edge1)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, edge1), (1, 2, edge1)]);
        p2.add_line(0, vec![(0, 1, edge2)]);
        p2.add_line(0, vec![(0, 1, edge3), (1, 6, edge1)]);

        let mut p3 = LinePattern::new();
        p3.require(0, 1);
        p3.add_line(0, vec![(0, 1, edge1)]);
        p3.add_line(0, vec![(0, 1, edge2)]);
        p3.add_line(0, vec![(0, 1, edge3)]);

        let builder: LineBuilder<_> = [p1.to_pattern(), p2.to_pattern(), p3.to_pattern()]
            .iter()
            .collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 23);
    }

    #[test]
    fn test_build_empty() {
        let p1 = LinePattern::<usize, usize, UnweightedEdge>::new();

        let builder: LineBuilder<_> = [p1.to_pattern()].iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 1);
    }
}
