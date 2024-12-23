#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod automaton;
pub mod branch_selector;
pub mod concrete;
pub mod constraint;
pub mod constraint_tree;
pub mod indexing;
pub mod matcher;
pub mod pattern;
pub mod predicate;
pub mod utils;

pub use branch_selector::{BranchSelector, CreateBranchSelector, EvaluateBranchSelector};
pub use constraint::{Constraint, DetHeuristic};
pub use constraint_tree::{ConditionedPredicate, ConstraintTree, ToConstraintsTree};
pub use indexing::{BindMap, IndexedData, IndexingScheme};
pub use matcher::{
    ManyMatcher, NaiveManyMatcher, PatternFallback, PatternID, PatternMatch, PortMatcher,
    SinglePatternMatcher,
};
pub use pattern::{Pattern, PatternLogic};
pub use predicate::{ArityPredicate, ConstraintLogic, Predicate, PredicatePattern};

use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) type HashMap<K, V> = FxHashMap<K, V>;
pub(crate) type HashSet<T> = FxHashSet<T>;
