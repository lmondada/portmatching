use std::{
    collections::{BTreeSet, VecDeque},
    hash::{BuildHasher, Hash},
};

use itertools::{Either, Itertools};

use crate::{
    automaton::{ScopeAutomaton, StateID},
    patterns::{IterationStatus, LinePattern, PredicatesIter},
    predicate::{are_compatible_predicates, EdgePredicate, PredicateCompatibility, Symbol},
    EdgeProperty, HashMap, HashSet, NodeProperty, Universe,
};

pub struct LineBuilder<U: Universe, PNode, PEdge> {
    patterns: Vec<LinePattern<U, PNode, PEdge>>,
    // Map from ({pattern_id} x line_nb) to automaton state
    det_states: HashMap<(BTreeSet<usize>, usize), StateID>,
}

impl<U: Universe, PNode, PEdge> LineBuilder<U, PNode, PEdge> {
    #[allow(unused)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_patterns(patterns: Vec<LinePattern<U, PNode, PEdge>>) -> Self {
        Self {
            patterns,
            det_states: HashMap::default(),
        }
    }

    #[allow(unused)]
    pub fn add_pattern(&mut self, pattern: LinePattern<U, PNode, PEdge>) {
        self.patterns.push(pattern);
    }

    pub fn build(mut self) -> ScopeAutomaton<PNode, PEdge>
    where
        PNode: NodeProperty,
        PEdge: EdgeProperty,
        PEdge::OffsetID: Eq + Hash,
    {
        let mut matcher = ScopeAutomaton::<PNode, PEdge>::new();

        // Convert patterns to pattern in construction
        let patterns = self
            .patterns
            .iter()
            .enumerate()
            .map(|(i, p)| PatternInConstruction::new(p.edge_predicates(), i))
            .collect::<Vec<_>>();

        let mut to_insert = VecDeque::new();
        to_insert.push_back((matcher.root(), patterns));
        while let Some((state, patterns)) = to_insert.pop_front() {
            // Filter out done patterns
            let (patterns_stages, done_ids): (Vec<_>, Vec<_>) =
                patterns.into_iter().partition_map(|mut p| {
                    let stage = p.edges.traversal_stage();
                    match stage {
                        IterationStatus::Finished => Either::Right(p.pattern_id),
                        _ => Either::Left((p, stage)),
                    }
                });
            // Insert flags for done patterns
            for p in done_ids {
                matcher.add_match(state, p.into());
            }
            let (patterns, stages): (Vec<_>, Vec<_>) = patterns_stages.into_iter().unzip();
            // The current stage of the iteration is the minimum of all stages
            let Some(&current_stage) = stages.iter().min() else {
                continue;
            };
            // Compute the next transitions
            let transitions = if matches!(current_stage, IterationStatus::LeftOver(_)) {
                // In the leftover stage, only use det transitions
                // Store predicates as a map (stage -> predicates)
                let stage_patterns = stages
                    .into_iter()
                    .map(|stage| match stage {
                        IterationStatus::LeftOver(i) => i,
                        _ => unreachable!("finished was filtered out and skeleton is smaller"),
                    })
                    .zip(patterns)
                    .into_grouping_map()
                    .collect();
                self.only_det_predicates(state, stage_patterns)
            } else {
                self.get_compatible_predicates(state, patterns)
            };
            let new_children = self.add_transitions(&mut matcher, &transitions);

            // Enqueue new states
            for (new_state, mut new_transition) in new_children.into_iter().zip(transitions) {
                if let Some(new_state) = new_state {
                    if let Some(stage) = leftover_stage(&mut new_transition.patterns) {
                        // Insert into our catalog of deterministic states
                        let pattern_ids = new_transition
                            .patterns
                            .iter()
                            .map(|p| p.pattern_id)
                            .collect();
                        self.det_states
                            .entry((pattern_ids, stage))
                            .or_insert(new_state);
                    }
                    to_insert.push_back((new_state, new_transition.patterns));
                }
            }
        }
        matcher
    }

    fn add_transitions(
        &self,
        matcher: &mut ScopeAutomaton<PNode, PEdge>,
        transitions: &[TransitionInConstruction<'_, U, PNode, PEdge>],
    ) -> Vec<Option<StateID>>
    where
        PNode: NodeProperty,
        PEdge: EdgeProperty,
        PEdge::OffsetID: Eq,
    {
        let mut new_states = vec![None; transitions.len()];
        // Enumerate them so that we can restore their ordering
        let transitions = transitions.iter().enumerate();

        for (source, transitions) in partition_by(transitions, |(_, t)| t.source) {
            let (inds, targets, preds, scopes): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = transitions
                .into_iter()
                .map(|(i, t)| (i, t.target, t.pred.clone(), t.scope()))
                .multiunzip();

            // make source non-deterministic if necessary
            match are_compatible_predicates(&preds) {
                PredicateCompatibility::Deterministic => {}
                PredicateCompatibility::NonDeterministic => matcher.make_non_det(source),
                PredicateCompatibility::Incompatible => {
                    panic!("trying to insert non-compatible transitions");
                }
            }

            let added_states = matcher.set_children(source, preds, &targets, scopes);

            for (i, new_state) in inds.into_iter().zip(added_states) {
                new_states[i] = new_state;
            }
        }

        new_states
    }

    /// The next predicate of each pattern
    ///
    /// Tries to consume the first predicate of each pattern. If there is a mix
    /// of deterministic and non-deterministic transitions, then prioritise
    /// non-deterministic ones and bundle all deterministc ones behind a single
    /// True transition.
    fn get_compatible_predicates<'a>(
        &self,
        source: StateID,
        patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: NodeProperty,
        PEdge: EdgeProperty,
        PEdge::OffsetID: Hash,
    {
        let mut patterns = patterns
            .into_iter()
            .map(|mut p| (p.edges.peek().expect("Not finished").transition_type(), p))
            .into_group_map();
        let nondet_patterns = patterns.remove(&PredicateCompatibility::NonDeterministic);
        let det_patterns = patterns.remove(&PredicateCompatibility::Deterministic);
        debug_assert!(patterns.is_empty());

        let mut transitions = Vec::new();
        if let Some(patterns) = nondet_patterns {
            transitions.append(&mut create_transitions(
                source,
                patterns
                    .into_iter()
                    .map(|mut p| (p.next_edge().expect("Not finished"), p)),
            ));
        }
        if let Some(patterns) = det_patterns {
            if transitions.is_empty() {
                // No non-det, so we can just add all deterministic transitions
                transitions.append(&mut create_transitions(
                    source,
                    patterns
                        .into_iter()
                        .map(|mut p| (p.next_edge().expect("Not finished"), p)),
                ));
            } else {
                // Add a single True transition, and put all det transitions there
                transitions.push(TransitionInConstruction {
                    source,
                    target: None,
                    pred: EdgePredicate::True,
                    patterns,
                })
            }
        }
        transitions
    }

    /// Force deterministic transitions
    ///
    /// Unlike other transition functions, this does the patterns in the children
    /// states do not necessary form a partition: patterns might be cloned up
    /// to `n` times, yielding an exponential overhead if this is called repeatedly.
    fn only_det_predicates<'a, H: BuildHasher>(
        &self,
        source: StateID,
        mut patterns: std::collections::HashMap<
            usize,
            Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
            H,
        >,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: NodeProperty,
        PEdge: EdgeProperty,
        PEdge::OffsetID: Hash,
    {
        // Find the min stage
        let Some(&min_stage) = patterns.keys().min() else {
            return Vec::new();
        };
        // Split patterns between min stage and other stages
        let min_patterns = patterns.remove(&min_stage).unwrap();
        let mut other_patterns = patterns.into_values().flatten().collect_vec();

        // Introduce the transitions as normal for the minimal stage
        // We need to be careful about the "scope" of the patterns
        // If the scope is smaller than the total scope, then we need to add
        // a child node to handle the smaller scope.
        let min_scopes = min_patterns.iter().map(|p| p.scope.clone()).collect_vec();
        let scope = min_scopes.iter().max_by_key(|s| s.len()).unwrap();
        let (mut in_scope_patterns, mut out_scope_patterns): (Vec<_>, Vec<_>) = min_patterns
            .iter()
            .cloned()
            .partition(|p| scope.is_superset(&p.scope));
        if !out_scope_patterns.is_empty() {
            // we need to split the predicates into two halves, because
            // the scopes do not have a join
            in_scope_patterns.extend(other_patterns.iter().cloned());
            out_scope_patterns.extend(other_patterns.iter().cloned());
            return vec![
                TransitionInConstruction {
                    source,
                    target: None,
                    pred: EdgePredicate::True,
                    patterns: in_scope_patterns,
                },
                TransitionInConstruction {
                    source,
                    target: None,
                    pred: EdgePredicate::True,
                    patterns: out_scope_patterns,
                },
            ];
        }
        // We handle the predicates with maximal scope
        let min_predicates = in_scope_patterns
            .iter_mut()
            .map(|p| p.next_edge().expect("Not finished"))
            .collect_vec();
        let mut transitions =
            create_transitions(source, min_predicates.into_iter().zip(in_scope_patterns));
        // Then add all other patterns to each possible transition
        for t in &mut transitions {
            t.patterns.append(&mut other_patterns.clone());
        }

        // Finally, we add a Fail transition to all other stages together
        // Also add patterns with smaller scope
        other_patterns.extend(
            min_patterns
                .into_iter()
                // The scope is contained, so just check the size
                .filter(|p| p.scope.len() < scope.len()),
        );
        if !other_patterns.is_empty() {
            let next_state = self.try_reuse_det_state(&mut other_patterns);
            transitions.push(TransitionInConstruction {
                source,
                target: next_state,
                pred: EdgePredicate::Fail,
                patterns: other_patterns,
            });
        }
        transitions
    }

    /// Find out whether a state that we can reuse exists
    fn try_reuse_det_state(
        &self,
        patterns: &mut [PatternInConstruction<'_, U, PNode, PEdge>],
    ) -> Option<StateID>
    where
        PNode: NodeProperty,
        PEdge: EdgeProperty,
    {
        let Some(stage) = leftover_stage(patterns) else {
            panic!("Can only reuse states in the LeftOver stage")
        };
        let pattern_ids = patterns.iter().map(|p| p.pattern_id).collect();
        self.det_states.get(&(pattern_ids, stage)).copied()
    }
}

fn leftover_stage<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty>(
    patterns: &mut [PatternInConstruction<'_, U, PNode, PEdge>],
) -> Option<usize> {
    patterns
        .iter_mut()
        .map(|p| {
            if let IterationStatus::LeftOver(i) = p.edges.traversal_stage() {
                Some(i)
            } else {
                None
            }
        })
        .min()
        .flatten()
}

fn create_transitions<'a, U, PNode, PEdge>(
    source: StateID,
    iter: impl IntoIterator<
        Item = (
            EdgePredicate<PNode, PEdge, PEdge::OffsetID>,
            PatternInConstruction<'a, U, PNode, PEdge>,
        ),
    >,
) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
where
    U: Universe,
    PNode: NodeProperty,
    PEdge: EdgeProperty,
    PEdge::OffsetID: Hash,
{
    iter.into_iter()
        .into_group_map()
        .into_iter()
        .map(|(pred, patterns)| TransitionInConstruction {
            source,
            target: None,
            pred,
            patterns,
        })
        .collect()
}

#[derive(Clone)]
struct PatternInConstruction<'a, U: Universe, PNode, PEdge: EdgeProperty> {
    /// Predicates of the pattern yet to be inserted
    edges: PredicatesIter<'a, U, PNode, PEdge>,
    /// Pattern ID
    pattern_id: usize,
    /// The scope required to interpret the predicates
    /// This is initially empty, but is updated as predicates are consumed
    scope: HashSet<Symbol>,
}

struct TransitionInConstruction<'a, U: Universe, PNode, PEdge: EdgeProperty> {
    source: StateID,
    target: Option<StateID>,
    pred: EdgePredicate<PNode, PEdge, PEdge::OffsetID>,
    patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
}

impl<'a, U: Universe, PNode, PEdge: EdgeProperty> TransitionInConstruction<'a, U, PNode, PEdge> {
    fn scope(&self) -> HashSet<Symbol> {
        self.patterns
            .iter()
            .flat_map(|p| p.scope.iter().copied())
            .collect()
    }
}

impl<'a, U: Universe, PNode: Clone, PEdge: EdgeProperty>
    PatternInConstruction<'a, U, PNode, PEdge>
{
    fn new(edges: PredicatesIter<'a, U, PNode, PEdge>, pattern_id: usize) -> Self {
        Self {
            edges,
            pattern_id,
            scope: [Symbol::root()].into_iter().collect(),
        }
    }

    fn next_edge(&mut self) -> Option<EdgePredicate<PNode, PEdge, PEdge::OffsetID>>
    where
        PNode: NodeProperty,
    {
        let edge = self.edges.next()?;
        if let EdgePredicate::LinkNewNode { new_node, .. } = edge {
            self.scope.insert(new_node);
        }
        Some(edge)
    }
}

impl<U: Universe, PNode, PEdge> Default for LineBuilder<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            det_states: HashMap::default(),
        }
    }
}

impl<U: Universe, PNode, PEdge> FromIterator<LinePattern<U, PNode, PEdge>>
    for LineBuilder<U, PNode, PEdge>
{
    fn from_iter<T: IntoIterator<Item = LinePattern<U, PNode, PEdge>>>(iter: T) -> Self {
        Self {
            patterns: iter.into_iter().collect(),
            det_states: HashMap::default(),
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
    use crate::{patterns::LinePattern, EdgeProperty};

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
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 1)]);

        let builder: LineBuilder<_, _, _> = [p1, p2].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 7);
    }

    #[test]
    #[should_panic]
    fn test_incompatible_preds() {
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0), (0, 2, 1)]);

        let builder: LineBuilder<_, _, _> = [p1, p2].into_iter().collect();
        builder.build();
    }

    #[test]
    fn test_build_two_dir() {
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p1.add_line(0, vec![(0, 3, 1), (3, 4, 0)]);
        p1.add_line(1, vec![(1, 5, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p2.add_line(0, vec![(0, 3, 1), (3, 4, 0)]);
        p2.add_line(3, vec![(3, 5, 0)]);

        let builder: LineBuilder<_, _, _> = [p1, p2].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 18);
    }

    #[test]
    fn test_build_two_different_length() {
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p1.add_line(0, vec![(0, 3, 0), (3, 4, 0)]);
        p1.add_line(1, vec![(1, 5, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0)]);
        p2.add_line(0, vec![(0, 3, 0), (3, 4, 0)]);

        let builder: LineBuilder<_, _, _> = [p1, p2].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 15);
    }

    #[test]
    fn test_build_full_det() {
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]);
        p1.add_line(0, vec![(0, 1, 1), (1, 5, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p2.add_line(0, vec![(0, 1, 1)]);

        let builder: LineBuilder<_, _, _> = [p1, p2].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 13);
    }

    #[test]
    fn test_build_full_3_det() {
        let mut p1 = LinePattern::new();
        p1.require(0, 1);
        p1.add_line(0, vec![(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0)]);
        p1.add_line(0, vec![(0, 1, 1), (1, 5, 0)]);
        p1.add_line(0, vec![(0, 1, 2), (1, 6, 0)]);

        let mut p2 = LinePattern::new();
        p2.require(0, 1);
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p2.add_line(0, vec![(0, 1, 1)]);
        p2.add_line(0, vec![(0, 1, 2), (1, 6, 0)]);

        let mut p3 = LinePattern::new();
        p3.require(0, 1);
        p3.add_line(0, vec![(0, 1, 0)]);
        p3.add_line(0, vec![(0, 1, 1)]);
        p3.add_line(0, vec![(0, 1, 2)]);

        let builder: LineBuilder<_, _, _> = [p1, p2, p3].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 23);
    }

    #[test]
    fn test_build_empty() {
        let p1 = LinePattern::<usize, usize, usize>::new();

        let builder: LineBuilder<_, _, _> = [p1].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 1);
    }
}
