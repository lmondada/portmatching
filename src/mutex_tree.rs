//! Express logic relations between constraints as trees.
//!
//! The idea is to define the semantics of constraints by how they interact
//! with one another. Siblings in the tree model mutually exclusive constraints,
//! whereas a path from the root is a conjunction of constraints: the grandchildren
//! of a child thus represent further refinements of the child's constraint.
//!
//! More concretely, a constraint tree is a tree along with the following data:
//!   - tree edges are labelled with constraints
//!   - tree leaves may be labelled with one or more integers, to be interpreted as
//!     indices with respect to a list of constraints.
//!   - an ordering of the children for any node.
//!
//! Its interpretation is most easily explained by viewing the tree as a state
//! automaton, where we say that node `n` is reachable for a given input if
//!  - `n` is the root, or
//!  - the parent `p` of `n` is reachable and `n` is the smallest sibling for
//!    which the edge constraint from `p` to the sibling is satisfied.
//!
//! Then, a constraint tree encodes a list of constraints C[0], ... C[k] if the
//! following equivalence holds:
//!      C[i] is satisfied on input D
//!          <=>
//!      a state with label i in the constraint tree is reachable on input D.
//!
//! Note that siblings are mutually exclusive by definition: constraints A, B, C
//! on children of a node are interpreted as A, B ∧ ¬A, and C ∧ ¬A ∧ ¬B
//! respectively.

mod build;

use crate::Constraint;

/// Define the semantics of constraints using Constraint trees.
///
/// Provide a way to decompose a list of (arbitrary) constraints
/// into a tree of constraints. In the worst case, this is
/// always possible by choosing a single constraint and returning the tree with
/// one root and one leaf node, with the single edge labelled with the chosen
/// constraint.
///
/// The client is free to select which subset of constraints to include in the tree.
/// For good performance, it is however recommended that the client always
/// processes the "smallest" constraints first, according to some total order
/// of the constraints. This will ensure a maximum overlap between different
/// patterns in the final pattern matching data structure.
pub trait ToConstraintsTree<K>
where
    Self: Sized,
{
    /// Organise a list of constraints into a tree of constraints.
    ///
    /// Node indices must be in [0, constraints.len()). Not all indices must be
    /// present in the tree. If so the tree is interpreted as the constraint tree
    /// of the subset of constraints present in the tree.
    fn to_constraints_tree(
        constraints: Vec<Constraint<K, Self>>,
    ) -> MutuallyExclusiveTree<Constraint<K, Self>>;
}

/// Condition predicate on a set of satisfied predicates.
pub trait ConditionedPredicate<K>: ToConstraintsTree<K>
where
    Self: Sized,
{
    /// A possibly simplified version of the constraint that is equivalent to
    /// `self` under the assumption that the predicates in `satisfied` are
    /// satisfied.
    fn conditioned(
        constraint: &Constraint<K, Self>,
        satisfied: &[&Constraint<K, Self>],
    ) -> Option<Constraint<K, Self>>;
}

/// The constraint tree datastructure expected by `ToConstraintsTree`.
///
/// The constraints at the root are interpreted as mutually exclusive, i.e.
/// constraints A, B, C on children of root are interpreted as A, B ∧ ¬A, and
/// C ∧ ¬A ∧ ¬B respectively.
/// All constraints will be evaluated on the same set of bindings, with a binding
/// provided for all index keys used by the constraints.
///
/// If a index label appears at least once, then it is assumed that the
/// constraint is satisfied exactly when a labelled state is reacheable.
#[derive(Clone, Debug)]
pub struct MutuallyExclusiveTree<C> {
    nodes: Vec<MutExTreeNode<C>>,
}

impl<C> MutuallyExclusiveTree<C> {
    /// Construct a new constraint tree with a root node.
    pub fn new() -> Self {
        let root = MutExTreeNode::new();
        Self { nodes: vec![root] }
    }

    /// Construct a new constraint tree that has depth one.
    ///
    /// Each element in `children` is a child of the root, with constraint
    /// indices given by the second element of the tuple.
    pub fn with_children(children: impl IntoIterator<Item = (C, Vec<usize>)>) -> Self
    where
        C: PartialEq,
    {
        let mut tree = Self::new();
        for (child, indices) in children {
            let child_index = tree.get_or_add_child(tree.root(), child);
            for index in indices {
                tree.add_constraint_index(child_index, index);
            }
        }
        tree
    }

    /// Get the index of the root node.
    pub fn root(&self) -> usize {
        0
    }

    /// Get the indices of the constraints at a node.
    pub fn constraint_indices(&self, node: usize) -> &[usize] {
        &self.nodes[node].constraint_indices
    }

    /// The set of constraints at node `node`.
    pub fn children(&self, node: usize) -> impl Iterator<Item = (usize, &C)> {
        self.nodes[node]
            .children
            .iter()
            .map(|child| (child.node_index, &child.constraint))
    }

    /// Add children to a node in the tree.
    pub fn add_children<'a>(
        &'a mut self,
        node: usize,
        constraints: impl IntoIterator<Item = C> + 'a,
    ) -> impl Iterator<Item = usize> + 'a
    where
        C: PartialEq,
    {
        constraints
            .into_iter()
            .map(move |c| self.get_or_add_child(node, c))
    }

    /// Get or add a child to a node in the tree.
    ///
    /// Returns the index of the child node. If the constraint does not exist,
    /// it is added and the index of the new node is returned. Otherwise the
    /// index of the existing node is returned.
    pub fn get_or_add_child(&mut self, node: usize, constraint: C) -> usize
    where
        C: PartialEq,
    {
        if self.nodes.len() <= node {
            panic!("Cannot add child to node that does not exist");
        }
        for child in &self.nodes[node].children {
            if child.constraint == constraint {
                return child.node_index;
            }
        }
        let child_index = self.nodes.len();
        self.nodes.push(MutExTreeNode::new());
        self.nodes[node].children.push(MutExTreeNodeChild {
            constraint,
            node_index: child_index,
        });
        child_index
    }

    /// Set the constraint index for a node in the tree.
    pub fn add_constraint_index(&mut self, node: usize, index: usize) {
        self.nodes[node].constraint_indices.push(index);
    }

    /// The number of nodes in the tree.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The number of children of a node.
    pub fn n_children(&self, node: usize) -> usize {
        self.nodes[node].children.len()
    }
}

impl<C> Default for MutuallyExclusiveTree<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C> MutExTreeNode<C> {
    fn new() -> Self {
        Self {
            constraint_indices: vec![],
            children: vec![],
        }
    }
}

/// A node in a mutually exclusive tree.
///
/// The `constraint_indices` is the (possibly empty) list of the indices
/// in the list of constraints associated with the tree.
#[derive(Clone, Debug)]
struct MutExTreeNode<P> {
    constraint_indices: Vec<usize>,
    children: Vec<MutExTreeNodeChild<P>>,
}

/// Pointer to child node in a mutually exclusive tree.
///
/// Pointing is done using an index into the list of nodes in the tree.
#[derive(Clone, Debug)]
struct MutExTreeNodeChild<C> {
    constraint: C,
    node_index: usize,
}
