use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    hash::Hash,
};

use itertools::Itertools;

use crate::{
    automaton::{ScopeAutomaton, StateID, Symbol},
    patterns::{IterationStatus, LinePattern, PredicatesIter},
    predicate::{are_compatible_predicates, EdgePredicate, PredicateCompatibility},
    utils::SharedIter,
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
        let free_symbols = SharedIter::new(Symbol::gen_symbols());
        let mut patterns = self
            .patterns
            .iter()
            .enumerate()
            .map(|(i, p)| PatternInConstruction::new(p.edge_predicates(free_symbols.clone()), i))
            .collect::<Vec<_>>();

        // insert empty patterns at root
        patterns.retain(|p| {
            if p.is_empty() {
                matcher.add_match(matcher.root(), p.pattern_id.into());
                false
            } else {
                true
            }
        });
        let mut to_insert = VecDeque::new();
        to_insert.push_back((matcher.root(), patterns));
        while let Some((state, predicates)) = to_insert.pop_front() {
            // Partition children by predicate
            let transitions = self.next_transitions(state, predicates);
            let new_children = self.add_transitions(&mut matcher, &transitions);

            // Enqueue new states
            for (new_state, mut new_transition) in new_children.into_iter().zip(transitions) {
                // Remove finished patterns and record match
                new_transition.patterns.retain_mut(|p| {
                    if p.edges.traversal_stage() == IterationStatus::Finished {
                        matcher.add_match(
                            new_state.expect("An old state was final"),
                            p.pattern_id.into(),
                        );
                        false
                    } else {
                        true
                    }
                });
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

    fn next_transitions<'a, IS>(
        &self,
        source: StateID,
        predicates: Vec<PatternInConstruction<'a, U, PNode, PEdge, IS>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge, IS>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Clone + Iterator<Item = Symbol>,
    {
        let stages = partition_by(predicates, |p| p.edges.traversal_stage());
        debug_assert!(!stages.contains_key(&IterationStatus::Finished));

        if stages.is_empty() {
            Vec::new()
        } else if stages.keys().len() == 1 {
            let vals = stages.into_values().next().unwrap();
            println!("compatible case");
            self.compatible_transitions(source, vals)
        } else if stages
            .keys()
            .all(|s| matches!(s, IterationStatus::LeftOver(_)))
        {
            // In the leftover stage, only use det transitions
            println!("Only det case");
            self.only_det_transitions(source, stages)
        } else {
            // Add a non-deterministic intermediate state
            println!("non-det");
            self.non_det_transitions(source, stages)
        }
    }

    fn add_transitions<IS>(
        &self,
        matcher: &mut ScopeAutomaton<PNode, PEdge>,
        transitions: &[TransitionInConstruction<'_, U, PNode, PEdge, IS>],
    ) -> Vec<Option<StateID>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Iterator,
        IS::Item: Clone,
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

    /// All transitions from `source` for `patterns`
    ///
    /// All predicates must be mutually exclusive, otherwise this panics.
    fn compatible_transitions<'a, IS>(
        &self,
        source: StateID,
        patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge, IS>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge, IS>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Iterator<Item = Symbol>,
    {
        let partition = partition_by(patterns, |p| p.edges.next().expect("not finished"));
        partition
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
    fn only_det_transitions<'a, IS>(
        &self,
        source: StateID,
        patterns: HashMap<IterationStatus, Vec<PatternInConstruction<'a, U, PNode, PEdge, IS>>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge, IS>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Clone + Iterator<Item = Symbol>,
    {
        // Partition between the min stage and the others
        let Some(&min_stage) = patterns.keys().min_by_key(|s| match s {
            IterationStatus::LeftOver(s) => s,
            _ => unreachable!(),
        }) else {
            return Vec::new();
        };
        let mut patterns = coarser_partition(patterns, |&stage| stage == min_stage);
        let min_patterns = patterns.remove(&true).unwrap();
        let mut other_patterns = patterns.remove(&false).unwrap();

        // For the min stage, introduce the transitions (almost) as normal
        let mut preds = Vec::new();
        for (pred, mut patterns) in
            partition_by(min_patterns, |p| p.edges.next().expect("not finished"))
        {
            // Note that we clone all non-minial patterns, as they might match
            // these transitions too
            patterns.extend(other_patterns.iter().cloned());
            preds.push(TransitionInConstruction {
                source,
                target: None,
                pred,
                patterns,
            });
        }

        // We bucket all other stages together
        let next_state = self.try_reuse_det_state(&mut other_patterns);
        preds.push(TransitionInConstruction {
            source,
            target: next_state,
            pred: EdgePredicate::True {
                line: usize::MAX,
                deterministic: true,
            },
            patterns: other_patterns,
        });
        preds
    }

    /// Find out whether a state that we can reuse exists
    fn try_reuse_det_state<IS>(
        &self,
        patterns: &mut [PatternInConstruction<'_, U, PNode, PEdge, IS>],
    ) -> Option<StateID>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Clone + Iterator<Item = Symbol>,
    {
        let Some(stage) = leftover_stage(patterns) else {
            panic!("Can only reuse states in the LeftOver stage")
        };
        let pattern_ids = patterns.iter().map(|p| p.pattern_id).collect();
        self.det_states.get(&(pattern_ids, stage)).copied()
    }

    fn non_det_transitions<'a, IS>(
        &self,
        source: StateID,
        patterns: HashMap<IterationStatus, Vec<PatternInConstruction<'a, U, PNode, PEdge, IS>>>,
    ) -> Vec<TransitionInConstruction<'a, U, PNode, PEdge, IS>>
    where
        PNode: Copy + Eq + Hash,
        PEdge: Copy + Eq + Hash,
        IS: Clone + Iterator<Item = Symbol>,
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
    }
}

fn leftover_stage<U: Universe, PNode: Copy, PEdge: Copy, IS: Iterator<Item = Symbol>>(
    patterns: &mut [PatternInConstruction<'_, U, PNode, PEdge, IS>],
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
struct PatternInConstruction<'a, U: Universe, PNode, PEdge, IS> {
    edges: PredicatesIter<'a, U, PNode, PEdge, IS, Symbol>,
    pattern_id: usize,
}

struct TransitionInConstruction<'a, U: Universe, PNode, PEdge, IS> {
    source: StateID,
    target: Option<StateID>,
    pred: EdgePredicate<PNode, PEdge, Symbol>,
    patterns: Vec<PatternInConstruction<'a, U, PNode, PEdge, IS>>,
}

impl<'a, U: Universe, PNode: Copy, PEdge: Copy, IS> PatternInConstruction<'a, U, PNode, PEdge, IS> {
    fn new(edges: PredicatesIter<'a, U, PNode, PEdge, IS, Symbol>, pattern_id: usize) -> Self {
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
        assert_eq!(matcher.n_states(), 5);
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
        assert_eq!(matcher.n_states(), 12);
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
        assert_eq!(matcher.n_states(), 12);
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
        assert_eq!(matcher.n_states(), 10);
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
        assert_eq!(matcher.n_states(), 18);
    }

    #[test]
    fn test_build_empty() {
        let p1 = LinePattern::<usize, usize, usize>::new();

        let builder: LineBuilder<_, _, _> = [p1].into_iter().collect();
        let matcher = builder.build();
        assert_eq!(matcher.n_states(), 1);
    }
}
