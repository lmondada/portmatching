#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

pub mod automaton;
pub mod concrete;
pub mod constraint;
pub mod indexing;
pub mod matcher;
pub mod pattern;
pub mod utils;

pub use constraint::{
    ArityPredicate, ConditionalPredicate, Constraint, ConstraintEvaluator, ConstraintTag,
    EvaluateConstraints, EvaluatePredicate, Tag,
};
pub use indexing::{BindMap, IndexedData, IndexingScheme};
pub use matcher::{
    ManyMatcher, NaiveManyMatcher, PatternFallback, PatternID, PatternMatch, PortMatcher,
    SinglePatternMatcher,
};
pub use pattern::{PartialPattern, Pattern};

use rustc_hash::{FxHashMap, FxHashSet};

pub(crate) type HashMap<K, V> = FxHashMap<K, V>;
pub(crate) type HashSet<T> = FxHashSet<T>;
