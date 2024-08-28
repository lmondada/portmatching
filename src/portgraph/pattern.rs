//! Patterns for matching port graphs.

use crate::{pattern::ConcretePattern, utils::is_connected, Pattern};
use portgraph::{LinkView, NodeIndex, PortGraph};

use super::constraint::{constraint_vec, PGConstraint, PGPatternError};

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
    type Constraint = PGConstraint;
    type Error = PGPatternError;

    fn try_to_constraint_vec(&self) -> Result<Vec<Self::Constraint>, Self::Error> {
        constraint_vec(&self.graph, self.root.ok_or(PGPatternError::NoRoot)?)
    }
}

impl ConcretePattern for PGPattern<PortGraph> {
    type Host = PortGraph;

    fn as_host(&self) -> &Self::Host {
        &self.graph
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
