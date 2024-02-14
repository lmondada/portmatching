use std::{hash::Hash};

use crate::{BiMap, HashSet};

pub type ScopeMap<C: ScopeConstraint> = BiMap<C::Symbol, C::Value>;
pub type Scope<C: ScopeConstraint> = HashSet<C::Symbol>;

// /// The variance of a scope constraint
// ///
// /// Determines how a scope constraint changes with taking subsets or supersets
// /// of scopes:
// pub enum Variance {
//     /// If the constraint is satisfied on a scope, it is also satisfied on any
//     /// superset of the scope,
//     Covariant,
//     /// If the constraint is satisfied on a scope, it is also satisfied on any
//     /// subset of the scope (that contains the constraint scope),
//     Contravariant,
//     /// The constraint is satisfied on any subset or superset of the scope (that
//     /// contains the constraint scope).
//     Invariant,
// }

/// Split constraints into sets
///
/// Split the constraints into smaller sets of constraints, each set being
/// labelled by a constraint to discriminate them. This split is used
/// recursively to determine the automaton transitions.
///
/// Transitions will either be labelled with a constraint or
/// treated as epsilon transitions, i.e. the transition can be taken without
/// any constraint.
///
/// In the case of deterministic partitions, only the last transition may be
/// unlabelled. This is the last-resort "FAIL" transition.
///
/// Repeated partitioning of constraint sets creates a hierarchy of constraints,
/// corresponding to the state graph of the automaton.
///
/// ## Advice
/// Any partition will be a valid non-deterministic partition but the number
/// of non-deterministic partitions should be minimised as the resulting
/// automata will require exponential runtime. When no deterministic partition
/// is possible, choose non-deterministic partitions that will result in
/// deterministic partitions later on.
pub struct ConstraintSplit<'r, C> {
    /// For the split to be deterministic, either the constraints in different
    /// sets must be mutually exclusive, or some constraints
    /// must be cloned to several sets. Hence deterministic transitions can be used to select between the
    /// constraint sets. Otherwise, constraints in different sets may be
    /// satisfied simultaneously. All resulting transitions will be non-deterministic.
    pub deterministic: bool,
    /// The constraints in the split, useful to derefence the indices
    pub constraints: &'r [C],
    /// An ordered list of constraint sets, given as constraint indices and whether
    /// the first constraint was consumed.
    pub constraint_sets: Vec<HashSet<(usize, bool)>>,
    /// The labels corresponding to the constraint sets
    /// The length of this vector is equal to the length of `constraint_sets`.
    pub labels: Vec<Option<C>>,
}

/// A symbol in a scope
pub trait Symbol: Copy + Eq + Hash + Ord {
    fn root() -> Self;
}

/// A constraint that can be checked on a scope
pub trait ScopeConstraint
where
    Self: Sized,
{
    type Symbol: Symbol;
    type Value: Copy + Eq + Hash;
    type DataRef<'a>: Copy;

    /// The scope that the constraint is checked on
    fn scope(&self) -> Scope<Self>;

    /// New symbols introduced in scope by this constraint
    fn new_symbols(&self) -> Scope<Self>;

    /// Check if the constraint is satisfied
    ///
    /// It is guaranteed that the scope map has values for all the elements
    /// in the scope.
    ///
    /// Return new symbol-value pairs to be added to the scope for all
    /// the new symbols.
    fn is_satisfied<'d>(&self, input: Self::DataRef<'d>, scope: &ScopeMap<Self>) -> Option<ScopeMap<Self>>;

    /// Split constraint set into smaller sets
    ///
    /// At each call to this function, progress must be made, i.e. a
    /// partition of the input set in at least two non-empty sets must be
    /// provided.
    ///
    /// See [`ConstraintPartition`] for more information on the valid choices
    /// of partitioning.
    fn split<'a>(constraints: impl Iterator<Item = &'a Self>) -> ConstraintSplit<'a, Self>;

    /// A unique ID for the target state of the constraint
    ///
    /// This is used to reuse states in the automaton.
    fn uid(&self) -> Option<String>;
}
