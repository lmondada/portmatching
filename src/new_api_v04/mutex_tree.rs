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
/// any variable assignent, only one of the constraints is satisfied. All
/// assign predicates must have the same variable RHS, and a constraint is
/// satisfied if one of the variable assignments returned by `check_assign`
/// corresponds to the assignment in the scope assignment.
/// In other words, the sets of variable assignments returned by the assign
/// predicates must be disjoint.
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

    /// Add children to a node in the tree.
    pub fn add_children(&mut self, node: usize, predicates: impl IntoIterator<Item = P>) {
        for p in predicates.into_iter() {
            self.add_child(node, p);
        }
    }

    /// Add a child to a node in the tree.
    pub fn add_child(&mut self, node: usize, predicate: P) {
        if self.nodes.len() <= node {
            panic!("Cannot add child to node that does not exist");
        }
        let child_index = self.nodes.len();
        self.nodes.push(MutExTreeNode::new());
        self.nodes[node].children.push(MutExTreeNodeChild {
            predicate,
            node_index: child_index,
        });
    }

    /// Set the constraint index for a node in the tree.
    pub fn set_constraint_index(&mut self, node: usize, index: usize) {
        self.nodes[node].constraint_index = Some(index);
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

/// A node in a mutually exclusive tree.
///
/// The `constraint_index` is the index of the constraint in the list of
/// constraints associated with the tree.
struct MutExTreeNode<P> {
    constraint_index: Option<usize>,
    children: Vec<MutExTreeNodeChild<P>>,
}

/// Pointer to child node in a mutually exclusive tree.
///
/// Pointing is done using an index into the list of nodes in the tree.
struct MutExTreeNodeChild<P> {
    predicate: P,
    node_index: usize,
}
