//! Express lists of constraints as trees of constraints.
//!
//! The idea is to define the semantics of constraints by how they interact
//! with one another.
//!
//! A constraint tree is a tree along with the following data and properties:
//!  - data
//!     - tree edges are labelled with constraints
//!     - tree nodes may be labelled with an integer, to be interpreted
//!       as an index with respect to a list of constraints.
//!  - properties
//!     - for any edge with constraint label A, the constraint labels on children
//!       edges B must be such that B => A.
//!     - for a node n with index i, with constraints A1, ..., Ak on the edges
//!       of the path from n to the root, we have
//!                           C[i] <=> A1 ∧ A2 ∧ ... ∧ Ak
//!       where C[i] is the i-th constraint in the list of constraints associated
//!       with the tree.
//!     - index labels in nodes are unique.

use super::{
    constraint::Constraint,
    predicate::{ArityPredicate, Predicate},
};

/// A trait to define the semantics of predicate enums.
///
/// The client must provide a way to decompose a list of (arbitrary) constraints
/// into a mutually exclusive tree of predicates. In the worst case, this is
/// always possible by choosing a single constraint and returning the tree with
/// one root and one leaf node, with the single edge labelled with the chosen
/// constraint.
///
/// The client is free to select which subset of constraints to include in the tree.
/// For good performance, it is however recommended that the client always
/// processes the "smallest" constraints first, according to some total order
/// of the constraints. This will ensure a maximum overlap between different
/// patterns in the final pattern matching data structure.
pub trait ToMutuallyExclusiveTree
where
    Self: Sized,
{
    /// Structure a list of constraints into a mutually exclusive tree.
    fn to_mutually_exclusive_tree(preds: Vec<Self>) -> MutuallyExclusiveTree<Self>;
}

/// A constraint tree with mutually exclusive constraints on root.
///
/// A constraint tree is a valid mutually exclusive tree if the following
/// additional properties hold:
///  - all constraints on the edges outgoing from the root are mutually exclusive,
///  - there is at least one vertex with an index label.
///
/// A set of constraints are mutually exclusive if for any data input and for
/// any variable assignent, only one of the constraints is satisfied. More
/// precisely, one of the following must hold
///  a) all constraints are filter constraints, and only one constraint can be
///     satisfied for any input data.
///  b) all constraints are assign constraints, and i) they are all assignments
///     to the same variable and ii) the sets of variable assignments returned
///     by the assign constraints for any input are disjoint.
///
/// If a index label appears at least once, then it is assumed that the
/// constraint is satisfied exactly when a labelled state is reacheable.
pub struct MutuallyExclusiveTree<P> {
    nodes: Vec<MutExTreeNode<P>>,
}

impl<P> MutuallyExclusiveTree<P> {
    /// Construct a new mutually exclusive tree with a root node.
    pub fn new() -> Self {
        let root = MutExTreeNode::new();
        Self { nodes: vec![root] }
    }

    /// Get the index of the root node.
    pub fn root(&self) -> usize {
        0
    }

    /// Get the index of the constraint at a node.
    pub fn constraint_index(&self, node: usize) -> Option<usize> {
        self.nodes[node].constraint_index
    }

    /// The set of constraints at node `node`.
    pub fn children(&self, node: usize) -> impl Iterator<Item = (usize, &P)> {
        self.nodes[node]
            .children
            .iter()
            .map(|child| (child.node_index, &child.predicate))
    }

    /// Add children to a node in the tree.
    pub fn add_children<'a>(
        &'a mut self,
        node: usize,
        predicates: impl IntoIterator<Item = P> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        predicates.into_iter().map(move |p| self.add_child(node, p))
    }

    /// Add a child to a node in the tree.
    pub fn add_child(&mut self, node: usize, predicate: P) -> usize {
        if self.nodes.len() <= node {
            panic!("Cannot add child to node that does not exist");
        }
        let child_index = self.nodes.len();
        self.nodes.push(MutExTreeNode::new());
        self.nodes[node].children.push(MutExTreeNodeChild {
            predicate,
            node_index: child_index,
        });
        child_index
    }

    /// Set the constraint index for a node in the tree.
    pub fn set_constraint_index(&mut self, node: usize, index: usize) {
        self.nodes[node].constraint_index = Some(index);
    }
}

impl<P> Default for MutuallyExclusiveTree<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> MutExTreeNode<P> {
    fn new() -> Self {
        Self {
            constraint_index: None,
            children: vec![],
        }
    }
}

impl<V, U, AP, FP> MutuallyExclusiveTree<Constraint<V, U, AP, FP>>
where
    AP: ArityPredicate,
    FP: ArityPredicate,
{
    /// Check that the tree is well-formed.
    ///
    /// Currently, this checks that the constraints on the edges outgoing from
    /// the root are either all Assign or all Filter constraints.
    pub fn is_valid_tree(&self) -> bool {
        let root = self.root();
        let all_assign = self
            .children(root)
            .all(|(_, pred)| matches!(pred.predicate(), Predicate::Assign(_)));
        let all_filter = self
            .children(root)
            .all(|(_, pred)| matches!(pred.predicate(), Predicate::Filter(_)));
        all_assign || all_filter
    }
}

/// A node in a mutually exclusive tree.
///
/// The `constraint_index` is the index of the constraint in the list of
/// constraints associated with the tree.
#[derive(Clone, Debug)]
struct MutExTreeNode<P> {
    constraint_index: Option<usize>,
    children: Vec<MutExTreeNodeChild<P>>,
}

/// Pointer to child node in a mutually exclusive tree.
///
/// Pointing is done using an index into the list of nodes in the tree.
#[derive(Clone, Debug)]
struct MutExTreeNodeChild<P> {
    predicate: P,
    node_index: usize,
}
