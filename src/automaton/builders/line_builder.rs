use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    hash::Hash,
};

use itertools::{Either, Itertools};

use crate::{
    automaton::{ScopeAutomaton, StateID},
    patterns::{IterationStatus, LinePattern, PredicatesIter},
    predicate::{are_compatible_predicates, EdgePredicate, PredicateCompatibility},
    Universe,
};

pub(crate) struct LineBuilder<U: Universe, PNode, PEdge> {
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
            det_states: HashMap::new(),
        }
    }

    #[allow(unused)]
    pub fn add_pattern(&mut self, pattern: LinePattern<U, PNode, PEdge>) {
        self.patterns.push(pattern);
    }

    pub fn build(mut self) -> ScopeAutomaton<PNode, PEdge>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        let mut matcher = ScopeAutomaton::<PNode, PEdge>::new();

        // Convert patterns to pattern in construction
        let mut patterns = self
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
            // Note that all Skeleton stages must be equal
            let Some(current_stage) = stages.iter().copied().reduce(|acc, e| {
                if let (IterationStatus::Skeleton(i), IterationStatus::Skeleton(j)) = (acc, e) {
                    if i != j {
                        panic!("incompatible stages");
                    }
                }
                acc.min(e)
            }) else {
                // Nothing to do
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
                    .into_group_map();
                self.only_det_transitions(state, stage_patterns)
            } else {
                // Extract first predicate from each pattern
                self.compatible_transitions(state, patterns)
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

    /*fn next_transitions<'a>(
        &self,
        source: StateID,
        predicates: Vec<EdgePredicate<PNode, PEdge>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        assert!(are_compatible_predicates(&predicates));
        self.compatible_transitions(source, vals)
        } else if stages
            .keys()
            .all(|s| matches!(s, IterationStatus::LeftOver(_)))
        {
        } else {
            // Add a non-deterministic intermediate state
            self.non_det_transitions(source, stages)
        }
    }*/

    fn add_transitions(
        &self,
        matcher: &mut ScopeAutomaton<PNode, PEdge>,
        transitions: &[TransitionInConstruction<'_, U, PNode, PEdge>],
    ) -> Vec<Option<StateID>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        let mut new_states = vec![None; transitions.len()];
        // Enumerate them so that we can restore their ordering
        let transitions = transitions.into_iter().enumerate();

        for (source, transitions) in partition_by(transitions, |(_, t)| t.source) {
            let (inds, targets, preds): (Vec<_>, Vec<_>, Vec<_>) = transitions
                .into_iter()
                .map(|(i, t)| (i, t.target, t.pred))
                .multiunzip();

            // make source non-deterministic if necessary
            match are_compatible_predicates(&preds) {
                PredicateCompatibility::Deterministic => {}
                PredicateCompatibility::NonDeterministic => matcher.to_non_det(source),
                PredicateCompatibility::Incompatible => {
                    panic!("trying to insert non-compatible transitions");
                }
            }

            let added_states = matcher.set_children(source, &preds, &targets);

            for (i, new_state) in inds.into_iter().zip(added_states) {
                new_states[i] = new_state;
            }
        }

        new_states
    }

    /// Consumes first predicate from each pattern and groups them by predicate
    fn compatible_transitions<'a>(
        &self,
        source: StateID,
        mut patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        let predicates = patterns
            .iter_mut()
            .map(|p| p.edges.next().expect("Not finished"))
            .collect_vec();
        // Group patterns by predicate
        predicates
            .into_iter()
            .zip(patterns)
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

    /// Force deterministic transitions
    ///
    /// Unlike other transition functions, this does the patterns in the children
    /// states do not necessary form a partition: patterns might be cloned up
    /// to `n` times, yielding an exponential overhead if this is called repeatedly.
    fn only_det_transitions<'a>(
        &self,
        source: StateID,
        mut patterns: HashMap<usize, Vec<PatternInConstruction<'a, U, PNode, PEdge>>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        // Find the min stage
        let Some(&min_stage) = patterns.keys().min() else {
            return Vec::new();
        };
        // Split patterns between min stage and other stages
        let mut min_patterns = patterns.remove(&min_stage).unwrap();
        let mut other_patterns = patterns.into_values().flatten().collect_vec();

        // For the min stage, introduce the transitions (almost) as normal
        let min_predicates = min_patterns
            .iter_mut()
            .map(|p| p.edges.next().expect("Not finished"))
            .collect_vec();
        // Group patterns by predicate
        let mut transitions = min_predicates
            .into_iter()
            .zip(min_patterns)
            .into_group_map()
            .into_iter()
            .map(|(pred, mut patterns)| {
                patterns.append(&mut other_patterns.clone());
                TransitionInConstruction {
                    source,
                    target: None,
                    pred,
                    patterns,
                }
            })
            .collect_vec();

        // We bucket all other stages together
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
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        let Some(stage) = leftover_stage(patterns) else {
            panic!("Can only reuse states in the LeftOver stage")
        };
        let pattern_ids = patterns.iter().map(|p| p.pattern_id).collect();
        self.det_states.get(&(pattern_ids, stage)).copied()
    }

    /*fn non_det_transitions<'a>(
        &self,
        source: StateID,
        patterns: HashMap<IterationStatus, Vec<PatternInConstruction<'a, U, PNode, PEdge>>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
    {
        let patterns = coarser_partition(patterns, |stage| match stage {
            &IterationStatus::Skeleton(i) => i,
            IterationStatus::LeftOver(_) => usize::MAX,
            IterationStatus::Finished => panic!("finished patterns should not be in the map"),
        });
        patterns
            .into_iter()
            .map(|(pred, patterns)| TransitionInConstruction {
                source,
                target: None,
                pred: EdgePredicate::True {
                    line: pred,
                    deterministic: false,
                },
                patterns,
            })
            .collect()
    }*/
}

fn leftover_stage<U: Universe, PNode: Copy, PEdge: Copy>(
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

#[derive(Clone)]
struct PatternInConstruction<'a, U: Universe, PNode, PEdge> {
    edges: PredicatesIter<'a, U, PNode, PEdge>,
    pattern_id: usize,
}

struct TransitionInConstruction<'a, U: Universe, PNode, PEdge> {
    source: StateID,
    target: Option<StateID>,
    pred: EdgePredicate<PNode, PEdge>,
    patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge>>,
}

impl<'a, U: Universe, PNode: Copy, PEdge: Copy> PatternInConstruction<'a, U, PNode, PEdge> {
    fn new(edges: PredicatesIter<'a, U, PNode, PEdge>, pattern_id: usize) -> Self {
        Self { edges, pattern_id }
    }

    fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

impl<U: Universe, PNode, PEdge> Default for LineBuilder<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            det_states: HashMap::new(),
        }
    }
}

impl<U: Universe, PNode, PEdge> FromIterator<LinePattern<U, PNode, PEdge>>
    for LineBuilder<U, PNode, PEdge>
{
    fn from_iter<T: IntoIterator<Item = LinePattern<U, PNode, PEdge>>>(iter: T) -> Self {
        Self {
            patterns: iter.into_iter().collect(),
            det_states: HashMap::new(),
        }
    }
}

fn partition_by<V, U, F>(iter: impl IntoIterator<Item = V>, f: F) -> HashMap<U, Vec<V>>
where
    U: Eq + Hash,
    F: for<'a> Fn(&'a mut V) -> U,
{
    let mut partitions = HashMap::new();
    for mut v in iter {
        let u = f(&mut v);
        partitions.entry(u).or_insert_with(Vec::new).push(v);
    }
    partitions
}

/// Merge keys in a hashmap partition
fn coarser_partition<U, UU, V, F: for<'a> Fn(&'a U) -> UU>(
    partition: HashMap<U, Vec<V>>,
    f: F,
) -> HashMap<UU, Vec<V>>
where
    U: Eq + Hash,
    UU: Eq + Hash,
{
    let mut new_partition = HashMap::new();
    for (u, mut v) in partition.into_iter() {
        let new_u = f(&u);
        new_partition
            .entry(new_u)
            .or_insert_with(Vec::new)
            .append(&mut v);
    }
    new_partition
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::patterns::LinePattern;

    use super::LineBuilder;

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
        assert_eq!(matcher.n_states(), 6);
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
        assert_eq!(matcher.n_states(), 17);
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
        assert_eq!(matcher.n_states(), 14);
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
        assert_eq!(matcher.n_states(), 12);
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
        assert_eq!(matcher.n_states(), 20);
    }

    #[test]
    fn test_build_empty() {
        let p1 = LinePattern::<usize, usize, usize>::new();

        let builder: LineBuilder<_, _, _> = [p1].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 1);
    }
}
