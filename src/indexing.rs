//! Access pattern matching subject data through indexing schemes.
//!
//! Index keys live in a symbol alphabet K. At runtime, they are bound by an
//! [IndexingScheme] to values in a domain universe V.
//!
//! The bindings are stored and retrieved in a map-like struct that implements
//! the [IndexMap] trait.

use crate::{HashMap, HashSet};
use std::{borrow::Borrow, collections::BTreeMap, fmt::Debug, hash::Hash};
use thiserror::Error;

/// Index key type alias for indexing schemes.
pub type Key<S> = <<S as IndexingScheme>::BindMap as BindMap>::Key;
/// Index value type alias for indexing schemes.
pub type Value<S> = <<S as IndexingScheme>::BindMap as BindMap>::Value;

/// Access a data structure through a set of index keys.
///
/// This trait assigns values to a set of index keys that can be used to access
/// the underlying data. All predicates for pattern matching are expressed as
/// expressions over elements of indexed values.
///
/// By implementing this trait, you create a specific indexing scheme that acts as
/// a lens through which the pattern matching automaton accesses and interprets the data.
///
/// ## Generic Parameter
/// - `Data`: The underlying data structure to index.
pub trait IndexingScheme {
    /// The index key - value map used to store index bindings.
    type BindMap: BindMap;

    /// List required bindings for an index key.
    ///
    /// Return the list of index keys that must be bound before `key` can be
    /// bound. The order of the returned keys specifies the order in which the
    /// bindings must be considered.
    ///
    /// When binding values, this will be called recursively on missing bindings.
    /// Avoid therefore cyclic bindings dependencies (and at least one index key
    /// must be bindable without prior bindings).
    fn required_bindings(&self, key: &Key<Self>) -> Vec<Key<Self>>;

    /// List all missing bindings for an index key, in topological order.
    ///
    /// Includes `key` if it is not already in `known_bindings`.
    fn missing_bindings(
        &self,
        key: &Key<Self>,
        known_bindings: &HashSet<Key<Self>>,
    ) -> Vec<Key<Self>> {
        enum DfsState<K> {
            DfsEnter(K),
            DfsExit(K),
        }
        use DfsState::*;

        // DFS traversing missing keys, until all required keys are in `known_bindings`
        let mut missing_keys = Vec::new();
        let mut visited = HashSet::default();
        let mut stack = Vec::new();
        if !known_bindings.contains(key) {
            stack.push(DfsEnter(*key));
            visited.insert(*key);
        }
        while let Some(state) = stack.pop() {
            match state {
                DfsEnter(key) => {
                    stack.push(DfsExit(key));
                    for req_key in self.required_bindings(&key) {
                        if !known_bindings.contains(&req_key) && visited.insert(req_key) {
                            stack.push(DfsEnter(req_key));
                        }
                    }
                }
                DfsExit(key) => {
                    missing_keys.push(key);
                }
            }
        }
        missing_keys
    }

    /// List all missing bindings for a list of index keys, in topological order.
    fn all_missing_bindings(
        &self,
        keys: impl IntoIterator<Item = Key<Self>>,
        known_bindings: impl IntoIterator<Item = Key<Self>>,
    ) -> Vec<Key<Self>> {
        let mut missing_keys = Vec::new();
        let mut known_bindings: HashSet<_> = known_bindings.into_iter().collect();
        for key in keys {
            if !known_bindings.contains(&key) {
                let missing = self.missing_bindings(&key, &known_bindings);
                missing_keys.extend(missing.clone());
                known_bindings.extend(missing);
            }
        }
        missing_keys
    }
}

/// Index key type alias for a data type.
pub type DataKey<D> =
    <<<D as IndexedData>::IndexingScheme as IndexingScheme>::BindMap as BindMap>::Key;
/// Index value type alias for a data type.
pub type DataValue<D> =
    <<<D as IndexedData>::IndexingScheme as IndexingScheme>::BindMap as BindMap>::Value;
/// Index key-value map type alias for a data type.
pub type DataBindMap<D> = <<D as IndexedData>::IndexingScheme as IndexingScheme>::BindMap;

/// A data structure that can be accessed through an [IndexingScheme].
pub trait IndexedData {
    /// The indexing scheme used to access the data.
    type IndexingScheme: IndexingScheme;

    /// List all valid bindings for an index key.
    ///
    /// Return a list of valid bindings for the index key.
    ///
    /// If `known_bindings` does not contain all the required bindings or the
    /// binding is not possible, return the empty list.
    fn list_bind_options(
        &self,
        key: &Key<Self::IndexingScheme>,
        known_bindings: &DataBindMap<Self>,
    ) -> Vec<Value<Self::IndexingScheme>>;

    /// Return all ways to extend `bindings` by binding all keys in `new_keys`,
    /// in order.
    ///
    /// If `allow_incomplete` is true, also return bindings that do not bind all
    /// keys in `new_keys`.
    fn bind_all(
        &self,
        bindings: DataBindMap<Self>,
        new_keys: impl IntoIterator<Item = DataKey<Self>>,
        allow_incomplete: bool,
    ) -> Vec<DataBindMap<Self>> {
        let mut all_bindings = vec![bindings];

        // Bind one key at a time to every possible value
        for key in new_keys {
            let mut new_bindings = Vec::new();
            for bindings in all_bindings {
                if bindings.get(&key).is_none() {
                    // Key is not bound yet, try to bind it
                    let valid_bindings = self.list_bind_options(&key, &bindings);
                    if valid_bindings.is_empty() && allow_incomplete {
                        // Can't bind this key, but it might still be useful as-is
                        new_bindings.push(bindings);
                    } else {
                        for value in valid_bindings {
                            let mut bindings = bindings.clone();
                            let Ok(()) = bindings.bind(key, value) else {
                                continue;
                            };
                            new_bindings.push(bindings);
                        }
                    }
                } else {
                    new_bindings.push(bindings);
                }
            }
            all_bindings = new_bindings;
        }
        all_bindings
    }
}

/// A map-like trait for index key-value bindings.
pub trait BindMap: Default + Clone {
    /// Index keys used to access the data.
    type Key: IndexKey;
    /// Values of the indexed data.
    type Value: IndexValue;
    /// A reference to a value in the map
    type ValueRef<'a>: Borrow<Self::Value> + 'a
    where
        Self: 'a;

    /// Lookup a binding for an index key.
    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>>;

    /// Bind a value to an index key.
    ///
    /// Returns an error if attempting to bind a new value to an existing index
    /// key.
    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), BindVariableError>;

    /// Retain only the bindings for the given keys.
    fn retain_keys(&mut self, keys: &HashSet<Self::Key>) {
        let mut new_self = Self::default();
        for &key in keys {
            if let Some(val) = self.get(&key) {
                new_self.bind(key, val.borrow().clone()).unwrap();
            }
        }
        *self = new_self;
    }
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
    /// Trying to bind an invalid key
    #[error("Cannot bind to key: {key}")]
    InvalidKey {
        /// The index key that is invalid
        key: String,
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
pub trait IndexValue: Clone + PartialEq + Debug + Hash + Borrow<Self> {}

impl<T: Eq + Copy + Debug + Hash> IndexKey for T {}
impl<T: Clone + PartialEq + Debug + Hash + Borrow<Self>> IndexValue for T {}

impl<K: IndexKey + 'static, V: IndexValue + 'static> BindMap for HashMap<K, V> {
    type Key = K;
    type Value = V;
    type ValueRef<'a> = &'a V;

    fn get(&self, var: &K) -> Option<Self::ValueRef<'_>> {
        self.get(var)
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

    fn retain_keys(&mut self, keys: &HashSet<Self::Key>) {
        self.retain(|key, _| keys.contains(key));
    }
}

impl<K: IndexKey + Ord + 'static, V: IndexValue + 'static> BindMap for BTreeMap<K, V> {
    type Key = K;
    type Value = V;
    type ValueRef<'a> = &'a V;

    fn get(&self, var: &K) -> Option<Self::ValueRef<'_>> {
        self.get(var)
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

    fn retain_keys(&mut self, keys: &HashSet<Self::Key>) {
        self.retain(|key, _| keys.contains(key));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use itertools::Itertools;

    use super::*;

    #[derive(Clone, Debug)]
    pub(crate) struct TestIndexingScheme;

    pub(crate) struct TestData;

    impl IndexingScheme for TestIndexingScheme {
        type BindMap = HashMap<usize, usize>;

        fn required_bindings(&self, key: &Key<Self>) -> Vec<Key<Self>> {
            if *key == 0 {
                vec![]
            } else {
                vec![key - 1]
            }
        }
    }

    impl IndexedData for TestData {
        type IndexingScheme = TestIndexingScheme;

        fn list_bind_options(
            &self,
            key: &Key<TestIndexingScheme>,
            known_bindings: &<TestIndexingScheme as IndexingScheme>::BindMap,
        ) -> Vec<Value<TestIndexingScheme>> {
            if *key == 0 || known_bindings.get(&(key - 1)).is_some() {
                // All previous keys were assigned, we can (dummy) bind the key
                vec![*key]
            } else {
                // Require key - 1 to be bound first
                vec![]
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

        // The list of keys that must be bound before `key` can be bound
        let all_missing_keys = scheme.missing_bindings(&key, &HashSet::default());
        assert_eq!(all_missing_keys, (0..=key).collect_vec());

        // Test binding a new value
        let all_index_maps = TestData.bind_all(index_map, all_missing_keys.iter().copied(), false);
        let index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(index_map.get(&key), Some(&key));

        // Must have bound all values smaller than `key` as well
        assert_eq!(index_map.len(), key + 1);

        // Test binding the same value again (should succeed)
        let all_index_maps =
            TestData.bind_all(index_map.clone(), all_missing_keys.iter().copied(), false);
        let new_index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(new_index_map, index_map);

        // Test binding multiple values at the same time, with some other values
        // already preset
        let index_map = HashMap::from_iter([(3, 3)]);
        let all_missing_keys = scheme.all_missing_bindings([1, 4], [3]);
        assert_eq!(all_missing_keys, vec![0, 1, 4]);
        let all_index_maps = TestData.bind_all(index_map, all_missing_keys, false);
        let index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(
            index_map.into_keys().sorted().collect_vec(),
            vec![0, 1, 3, 4]
        );
    }
}
