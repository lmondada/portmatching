//! Patterns for matching port graphs.

use std::iter;

use super::{
    indexing::PGIndexKey, predicate::PGTag, ConstraintSelector, PGConstraint, PGPredicate,
};
use crate::{
    constraint::{ConstraintPattern, PartialConstraintPattern},
    utils::{is_connected, portgraph::line_partition},
    HashMap, Pattern,
};

use itertools::Itertools;
use petgraph::visit::EdgeCount;
use portgraph::{LinkView, NodeIndex, PortGraph, PortView};
use thiserror::Error;

/// A concrete port graph pattern.
///
/// This has no variables, and thus can be represented by a port graph directly.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PGPattern<G> {
    pub(super) graph: G,
    pub(super) root: Option<NodeIndex>,
}

impl Pattern for PGPattern<PortGraph> {
    type PartialPattern = PartialConstraintPattern<PGIndexKey, PGPredicate>;
    type Error = PGPatternError;

    // Type aliases
    type Key = PGIndexKey;
    type Predicate = PGPredicate;
    type Tag = PGTag;
    type Evaluator = ConstraintSelector;

    fn required_bindings(&self) -> Vec<Self::Key> {
        // TODO: this is not very efficient...
        self.clone()
            .try_into_constraint_pattern()
            .unwrap()
            .constraints()
            .iter()
            .flat_map(|c| c.required_bindings())
            .copied()
            .unique()
            .collect()
    }

    fn try_into_partial_pattern(self) -> Result<Self::PartialPattern, PGPatternError> {
        Ok(self.try_into_constraint_pattern()?.into())
    }
}

/// Converting a pattern to constraints requires setting a root first.
#[derive(Debug)]
pub struct NoRootFound;

impl<G> PGPattern<G> {
    /// Whether a root has been set.
    pub fn is_root_set(&self) -> bool {
        self.root.is_some()
    }

    /// Get the graph defining the pattern.
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Set the pattern root.
    pub fn set_root(&mut self, root: NodeIndex) {
        self.root = Some(root);
    }

    /// Construct a pattern from a port graph, without setting root.
    pub fn from_host(graph: G) -> Self {
        Self { graph, root: None }
    }

    /// Construct a pattern from a port graph, use the default root choice.
    pub fn from_host_pick_root(graph: G) -> Self
    where
        G: LinkView,
    {
        let mut p = Self { graph, root: None };
        p.pick_root().unwrap();
        p
    }

    /// Construct a pattern given a port graph and node as root.
    pub fn from_host_with_root(graph: G, root: NodeIndex) -> Self {
        Self {
            graph,
            root: Some(root),
        }
    }
}

impl PGPattern<PortGraph> {
    /// Convert the pattern to a [`ConstraintPattern`].
    pub fn try_into_constraint_pattern(
        self,
    ) -> Result<ConstraintPattern<PGIndexKey, PGPredicate>, PGPatternError> {
        Ok(ConstraintPattern::from_constraints(constraint_vec(
            &self.graph,
            self.root.ok_or(PGPatternError::NoRoot)?,
        )?))
    }
}

impl<G: LinkView> PGPattern<G> {
    /// Whether the pattern is connected.
    ///
    /// A pattern must always be connected for pattern matching.
    pub fn is_connected(&self) -> bool {
        is_connected(&self.graph)
    }

    /// Attempt to pick a valid root.
    pub fn pick_root(&mut self) -> Result<NodeIndex, NoRootFound> {
        let root = self.graph.nodes_iter().next().ok_or(NoRootFound)?;
        self.root = Some(root);
        Ok(root)
    }
}

/// Error type for pattern generation.
#[derive(Debug, Clone, Copy, Error)]
pub enum PGPatternError {
    /// No root node was provided.
    #[error("No root node was provided")]
    NoRoot,
}

fn constraint_vec(graph: &PortGraph, root: NodeIndex) -> Result<Vec<PGConstraint>, PGPatternError> {
    if graph.edge_count() == 0 {
        return Ok(vec![PGConstraint::try_new(
            PGPredicate::HasNodeWeight(()),
            vec![PGIndexKey::root(0)],
        )
        .unwrap()]);
    }
    let mut constraints = Vec::new();
    let mut node_to_key = HashMap::from_iter([(root, PGIndexKey::root(0))]);
    let mut node_to_root_ind = HashMap::from_iter([(root, 0)]);
    for line in line_partition(graph, root) {
        let root = graph.port_node(line[0].0).unwrap();
        let n_roots = node_to_root_ind.len();
        let root_index = *node_to_root_ind.entry(root).or_insert(n_roots);
        let root_offset = graph.port_offset(line[0].0).unwrap();
        for (i, (left, right)) in line.into_iter().enumerate() {
            let left_node = graph.port_node(left).unwrap();
            let left_port = graph.port_offset(left).unwrap();
            let right_node = graph.port_node(right).unwrap();
            let right_port = graph.port_offset(right).unwrap();
            let left_key = *node_to_key.get(&left_node).expect("unknown edge LHS");
            let right_key = if let Some(right_key) = node_to_key.get(&right_node) {
                *right_key
            } else {
                // Create a new key
                let key = PGIndexKey::AlongPath {
                    path_root: root_index,
                    path_start_port: root_offset,
                    path_length: i + 1,
                };
                // Ensure it does not clash with previously bound keys
                let args = iter::once(key)
                    .chain(node_to_key.values().copied())
                    .collect_vec();
                constraints.push(
                    PGConstraint::try_new(
                        PGPredicate::IsNotEqual {
                            n_other: node_to_key.len(),
                        },
                        args,
                    )
                    .unwrap(),
                );
                node_to_key.insert(right_node, key);
                key
            };
            constraints.push(
                PGConstraint::try_new(
                    PGPredicate::IsConnected {
                        left_port,
                        right_port,
                    },
                    vec![left_key, right_key],
                )
                .unwrap(),
            );
        }
    }
    if constraints.is_empty() {
        // We add one (dummy) constraint for the empty pattern, forcing
        // the matcher to bind the first character to a position in the
        // string when matched. An alternative would be to explicitly
        // disallow empty patterns.
        constraints.push(
            PGConstraint::try_new(
                PGPredicate::IsNotEqual { n_other: 0 },
                vec![PGIndexKey::root(0)],
            )
            .unwrap(),
        );
    }
    Ok(constraints)
}
