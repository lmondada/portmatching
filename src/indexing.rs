//! Access pattern matching subject data through indexing schemes.
//!
//! Index keys live in a symbol alphabet K. At runtime, they are bound by an
//! [IndexingScheme] to values in a domain universe V.
//!
//! The bindings are stored and retrieved in a map-like struct that implements
//! the [IndexMap] trait.

mod binding_builder;

use crate::HashMap;
use std::{fmt::Debug, hash::Hash};
use thiserror::Error;

use self::binding_builder::BindingBuilder;

/// Access a data structure through a set of index keys.
///
/// This trait assigns values to a set of index keys that can be used to access
/// the underlying data. All predicates for pattern matching are expressed as
/// expressions over elements of indexed values.
///
/// By implementing this trait, you create a specific indexing scheme that acts as
/// a lens through which the pattern matching automaton accesses and interprets the data.
///
/// ## Generic Parameters
/// - `Data`: The underlying data structure to index.
/// - `Value`: The type of values in the indexed data.
pub trait IndexingScheme<Data, Value> {
    /// The type of index keys used to access the data.
    type Key: IndexKey;

    /// List valid bindings for an index key.
    ///
    /// Return either a list of valid bindings for the index key or a list of
    /// other keys that must be bound in `known_bindings` before `key` can
    /// itself be bound.
    ///
    /// When binding values, this will be called recursively on missing bindings.
    /// Avoid therefore cyclic bindings dependencies and at least one index key
    /// must be assignable for an empty `known_values`.
    fn valid_bindings(
        &self,
        key: &Self::Key,
        known_bindings: &impl IndexMap<Self::Key, Value>,
        data: &Data,
    ) -> BindingOptions<Self::Key, Value>;

    /// Try to recursively bind values for all `keys`.
    ///
    /// Return all possible binding maps. Returned maps may not have bindings
    /// for all keys, and different maps may have different key sets.
    /// This happens when an empty valid bindings list was returned for a key.
    fn try_bind_all<S: IndexMap<Self::Key, Value> + Clone>(
        &self,
        known_bindings: S,
        keys: impl IntoIterator<Item = Self::Key>,
        data: &Data,
    ) -> Result<Vec<S>, BindVariableError> {
        let mut curr_bindings = vec![BindingBuilder::new(keys, known_bindings)];
        let mut final_bindings = Vec::new();
        while let Some(mut builder) = curr_bindings.pop() {
            let Some(key) = builder.top_missing_key() else {
                // All keys have been bound, we can return the current binding
                final_bindings.push(builder.finish());
                continue;
            };
            let binding_options = self.valid_bindings(&key, builder.bindings(), data);
            match binding_options {
                Ok(ValidBindings(values)) => {
                    if values.is_empty() {
                        // Cannot bind this key, exclude it from further consideration
                        builder.exclude_key(key);
                        curr_bindings.push(builder);
                    } else {
                        // Successful binding
                        curr_bindings.extend(builder.apply_bindings(key, values)?);
                    }
                }
                Err(MissingIndexKeys(new_missing_keys)) => {
                    if !builder.extend_missing_keys(new_missing_keys) {
                        // All missing keys are already excluded, so we exclude
                        // `key` and proceed
                        builder.exclude_key(key);
                    }
                    curr_bindings.push(builder);
                }
            }
        }
        Ok(final_bindings)
    }
}

/// The result of a call to [IndexingScheme::valid_bindings].
///
/// Either a list of valid bindings or a list of missing index keys.
#[derive(Debug, Clone)]
pub enum BindingOptions<K, V> {
    /// A list of valid bindings, in the order they should be considered
    ValidBindings(Vec<V>),
    /// Indicates that valid bindings cannot be found unless bindings are
    /// provided for the missing index keys.
    MissingIndexKeys(Vec<K>),
}

/// A map-like trait for index key-value bindings.
pub trait IndexMap<K, V>: Default + Clone {
    /// Lookup a binding for an index key.
    fn get(&self, var: &K) -> Option<&V>;

    /// Bind a value to an index key.
    ///
    /// Returns an error if attempting to bind a new value to an existing index
    /// key.
    fn bind(&mut self, var: K, val: V) -> Result<(), BindVariableError>;

    /// Iterate over all bindings in the map.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&K, &V)> + 'a
    where
        K: 'a,
        V: 'a;
}

/// Errors in creating index key-value bindings.
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum BindVariableError {
    /// A binding already exists for the index key.
    #[error(
        "Cannot bind existing index key {key} to value {curr_value}: \
        already bound to {new_value}"
    )]
    VariableExists {
        /// The index key that already exists
        key: String,
        /// The value that already exists for the index key
        curr_value: String,
        /// The value that is being bound
        new_value: String,
    },
}

/// A shortcut for types that can be used as index keys.
///
/// This is implemented for all types that implement [`Eq`], [`Hash`], [`Copy`]
/// and [`Debug`].
pub trait IndexKey: Eq + Copy + Hash + Debug {}

/// A shortcut for types that can be used as index values.
///
/// This is implemented for all types that implement [`Clone`], [`PartialEq`]
/// and [`Debug`].
pub trait IndexValue: Clone + PartialEq + Debug {}

impl<T: Eq + Copy + Debug + Hash> IndexKey for T {}
impl<T: Clone + PartialEq + Debug> IndexValue for T {}

impl<K: IndexKey, V: IndexValue> IndexMap<K, V> for HashMap<K, V> {
    fn get(&self, var: &K) -> Option<&V> {
        self.get(var)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&K, &V)> + 'a
    where
        K: 'a,
        V: 'a,
    {
        self.iter()
    }

    fn bind(&mut self, var: K, val: V) -> Result<(), BindVariableError> {
        let curr_val = self.get(&var);
        if curr_val.is_some() && curr_val != Some(&val) {
            return Err(BindVariableError::VariableExists {
                key: format!("{:?}", var),
                curr_value: format!("{:?}", curr_val),
                new_value: format!("{:?}", val),
            });
        }
        self.insert(var, val);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use itertools::Itertools;

    use super::*;

    #[derive(Clone, Debug)]
    pub(crate) struct TestIndexingScheme;

    impl IndexingScheme<(), usize> for TestIndexingScheme {
        type Key = usize;
        type Map = HashMap<Self::Key, usize>;

        fn valid_bindings(
            &self,
            key: &Self::Key,
            known_bindings: &impl IndexMap<Self::Key, usize>,
            (): &(),
        ) -> BindingResult<Self::Key, usize> {
            if *key == 0 {
                // Key 0 maps to 0
                Ok(ValidBindings(vec![*key]))
            } else if known_bindings.get(&(key - 1)).is_some() {
                // Thanks for providing key - 1, we map key to itself
                Ok(ValidBindings(vec![*key]))
            } else {
                // Require key - 1 to be bound first
                Err(MissingIndexKeys(vec![key - 1]))
            }
        }
    }

    impl Default for TestIndexingScheme {
        fn default() -> Self {
            Self
        }
    }

    #[test]
    fn test_bind_with_scheme() {
        let index_map = HashMap::default();
        let scheme = TestIndexingScheme;
        let key = 4;

        // Test binding a new value
        let index_map = scheme
            .try_bind_all(index_map, vec![key], &())
            .unwrap()
            .into_iter()
            .exactly_one()
            .unwrap();
        assert_eq!(index_map.get(&key), Some(&key));
        // Must have bound all values smaller than `key` as well
        assert_eq!(index_map.len(), key + 1);

        // Test binding the same value again (should succeed)
        assert!(scheme.try_bind_all(index_map, vec![key], &()).is_ok());

        // Test binding multiple values at the same time, with some other values
        // already preset
        let index_map = HashMap::from_iter([(3, 3)]);
        let index_map = scheme
            .try_bind_all(index_map, vec![1, 4], &())
            .unwrap()
            .into_iter()
            .exactly_one()
            .unwrap();
        assert_eq!(
            index_map.into_keys().sorted().collect_vec(),
            vec![0, 1, 3, 4]
        );
    }
}
