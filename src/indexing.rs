//! Access pattern matching subject data through indexing schemes.
//!
//! Index keys live in a symbol alphabet K. At runtime, they are bound by an
//! [IndexingScheme] to values in a domain universe V.
//!
//! The bindings are stored and retrieved in a map-like struct that implements
//! the [IndexMap] trait.

use crate::{HashMap, HashSet};
use rustc_hash::FxHasher;
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    hash::{Hash, Hasher},
};
use thiserror::Error;

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
    type BindMap: BindMap<Key = Self::Key, Value = Self::Value>;

    /// The key type used for indexing
    type Key: IndexKey;
    /// The value type associated with indexed items
    type Value: IndexValue;

    /// List required bindings for an index key.
    ///
    /// Return the list of index keys that must be bound before `key` can be
    /// bound. The order of the returned keys specifies the order in which the
    /// bindings must be considered.
    ///
    /// When binding values, this will be called recursively on missing bindings.
    /// Avoid therefore cyclic bindings dependencies (and at least one index key
    /// must be bindable without prior bindings).
    fn required_bindings(&self, key: &Self::Key) -> Vec<Self::Key>;

    /// List all missing bindings for an index key, in topological order.
    ///
    /// Includes `key` if it is not already in `known_bindings`.
    fn missing_bindings(
        &self,
        key: &Self::Key,
        known_bindings: &HashSet<Self::Key>,
    ) -> Vec<Self::Key> {
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
        keys: impl IntoIterator<Item = Self::Key>,
        known_bindings: impl IntoIterator<Item = Self::Key>,
    ) -> Vec<Self::Key> {
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

/// A data structure that can be accessed through an [IndexingScheme].
pub trait IndexedData<K: IndexKey> {
    /// The indexing scheme used to access the data.
    type IndexingScheme: IndexingScheme<BindMap = Self::BindMap, Key = K, Value = Self::Value>;
    /// The value type that bindings resolve to
    type Value: IndexValue;
    /// The map used to store key-to-value bindings
    type BindMap: BindMap<Key = K, Value = Self::Value>;

    /// List all valid bindings for an index key.
    ///
    /// Return a list of valid bindings for the index key.
    ///
    /// If `known_bindings` does not contain all the required bindings or the
    /// binding is not possible, return the empty list.
    fn bind_options(&self, key: &K, known_bindings: &Self::BindMap) -> Vec<Self::Value>;

    /// Return all ways to extend `bindings` by binding all keys in `new_keys`,
    /// in order.
    fn bind_all(
        &self,
        bindings: Self::BindMap,
        new_keys: impl IntoIterator<Item = K>,
    ) -> Vec<Self::BindMap> {
        let mut all_bindings = vec![bindings];

        // Bind one key at a time to every possible value
        for key in new_keys {
            let mut new_bindings = Vec::new();
            for mut bindings in all_bindings {
                if bindings.get_binding(&key).is_unbound() {
                    // Key is not bound yet, try to bind it
                    let valid_bindings = self.bind_options(&key, &bindings);
                    if valid_bindings.is_empty() {
                        // Mark the key as impossible to bind
                        bindings.bind_failed(key);
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

#[derive(Debug, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
/// The result of getting the binding for a key
pub enum Binding<V> {
    /// Key is bound to a specific value
    Bound(V),
    /// No value has been assigned to the key yet.
    Unbound,
    /// A previous attempt at binding this key was unsuccessful.
    Failed,
}

impl<VRef> Binding<VRef> {
    /// Create a new Binding by borrowing the inner value
    ///
    /// # Arguments
    /// * `v` - The value to borrow
    pub fn borrowed<V>(&self) -> Binding<&V>
    where
        VRef: Borrow<V>,
    {
        match self {
            Binding::Bound(v) => Binding::Bound(v.borrow()),
            Binding::Failed => Binding::Failed,
            Binding::Unbound => Binding::Unbound,
        }
    }

    /// Create a new Binding by copying the inner value
    pub fn copied<V: Copy>(&self) -> Binding<V>
    where
        VRef: Borrow<V>,
    {
        match self {
            Binding::Bound(v) => Binding::Bound(*v.borrow()),
            Binding::Failed => Binding::Failed,
            Binding::Unbound => Binding::Unbound,
        }
    }
}

impl<V> Binding<V> {
    /// Whether this binding is in the Unbound state
    pub fn is_unbound(&self) -> bool {
        matches!(self, Binding::Unbound)
    }

    /// Whether this binding is in the Failed state
    pub fn is_failed(&self) -> bool {
        matches!(self, Binding::Failed)
    }

    /// Whether this binding is in the Bound state
    pub fn is_bound(&self) -> bool {
        matches!(self, Binding::Bound(_))
    }

    /// A reference to the bound value if this binding is Bound
    pub fn as_ref(&self) -> Binding<&V> {
        match self {
            Binding::Bound(v) => Binding::Bound(v),
            Binding::Failed => Binding::Failed,
            Binding::Unbound => Binding::Unbound,
        }
    }

    /// Map a function over the bound value if this binding is Bound
    ///
    /// # Arguments
    /// * `f` - The function to apply to the bound value
    pub fn map<U>(self, f: impl FnOnce(V) -> U) -> Binding<U> {
        match self {
            Binding::Bound(v) => Binding::Bound(f(v)),
            Binding::Failed => Binding::Failed,
            Binding::Unbound => Binding::Unbound,
        }
    }
}

/// A map-like trait for index key-value bindings.
pub trait BindMap: Default + Clone {
    /// The key type used for binding
    type Key: IndexKey;
    /// The value type that can be bound to keys
    type Value: IndexValue;

    /// Lookup a binding for an index key.
    fn get_binding(&self, var: &Self::Key) -> Binding<impl Borrow<Self::Value> + '_>;

    /// Bind a value to an index key.
    ///
    /// Returns an error if attempting to bind a new value to an existing index
    /// key.
    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), BindVariableError>;

    /// Mark an index key as unable to be bound
    fn bind_failed(&mut self, var: Self::Key);

    /// Retain only the bindings for the given keys.
    fn retain_keys(&mut self, keys: &BTreeSet<Self::Key>) {
        let mut new_self = Self::default();
        for &key in keys {
            match self.get_binding(&key) {
                Binding::Bound(val) => new_self.bind(key, val.borrow().clone()).unwrap(),
                Binding::Failed => new_self.bind_failed(key),
                Binding::Unbound => (),
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
pub trait IndexKey: Eq + Copy + Hash + Ord + Debug + 'static {}

/// A shortcut for types that can be used as index values.
///
/// This is implemented for all types that implement [`Clone`], [`PartialEq`]
/// and [`Debug`].
pub trait IndexValue: Clone + PartialEq + Debug + Hash + Borrow<Self> {}

impl<T: Eq + Copy + Ord + Debug + Hash + 'static> IndexKey for T {}
impl<T: Clone + PartialEq + Debug + Hash + Borrow<Self>> IndexValue for T {}

impl<K: IndexKey + 'static, V: IndexValue + 'static> BindMap for HashMap<K, Option<V>> {
    type Key = K;
    type Value = V;

    fn get_binding(&self, var: &K) -> Binding<impl Borrow<V> + '_> {
        match self.get(var) {
            Some(Some(v)) => Binding::Bound(v),
            Some(None) => Binding::Failed,
            None => Binding::Unbound,
        }
    }

    fn bind(&mut self, var: K, val: V) -> Result<(), BindVariableError> {
        let curr_val = self.get(&var);
        if curr_val.is_some() && curr_val.unwrap().as_ref() != Some(&val) {
            return Err(BindVariableError::VariableExists {
                key: format!("{:?}", var),
                curr_value: format!("{:?}", curr_val),
                new_value: format!("{:?}", val),
            });
        }
        self.insert(var, Some(val));
        Ok(())
    }

    fn bind_failed(&mut self, var: Self::Key) {
        self.insert(var, None);
    }

    fn retain_keys(&mut self, keys: &BTreeSet<Self::Key>) {
        self.retain(|key, _| keys.contains(key));
    }
}

impl<K: IndexKey + Ord + 'static, V: IndexValue + 'static> BindMap for BTreeMap<K, Option<V>> {
    type Key = K;
    type Value = V;

    fn get_binding(&self, var: &K) -> Binding<impl Borrow<V> + '_> {
        match self.get(var) {
            Some(Some(v)) => Binding::Bound(v),
            Some(None) => Binding::Failed,
            None => Binding::Unbound,
        }
    }

    fn bind(&mut self, var: K, val: V) -> Result<(), BindVariableError> {
        let curr_val = self.get(&var);
        if curr_val.is_some() && curr_val.unwrap().as_ref() != Some(&val) {
            return Err(BindVariableError::VariableExists {
                key: format!("{:?}", var),
                curr_value: format!("{:?}", curr_val),
                new_value: format!("{:?}", val),
            });
        }
        self.insert(var, Some(val));
        Ok(())
    }

    fn bind_failed(&mut self, var: Self::Key) {
        self.insert(var, None);
    }

    fn retain_keys(&mut self, keys: &BTreeSet<Self::Key>) {
        self.retain(|key, _| keys.contains(key));
    }
}

pub(crate) fn bindings_hash<S: BindMap>(
    bindings: &S,
    scope: impl IntoIterator<Item = S::Key>,
) -> u64 {
    let mut hasher = FxHasher::default();
    for key in scope {
        let value = bindings.get_binding(&key);
        value.as_ref().map(|v| v.borrow()).hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
pub(crate) mod tests {
    use itertools::Itertools;

    use crate::constraint::tests::TestKey;

    use super::*;

    #[derive(Clone, Debug)]
    pub(crate) struct TestStrIndexingScheme;

    #[derive(Clone, Debug)]
    pub(crate) struct TestUsizeIndexingScheme;

    pub(crate) struct TestData;

    impl IndexingScheme for TestStrIndexingScheme {
        type BindMap = HashMap<TestKey, Option<usize>>;

        type Key = TestKey;

        type Value = usize;

        fn required_bindings(&self, _: &Self::Key) -> Vec<Self::Key> {
            vec![]
        }
    }

    impl IndexingScheme for TestUsizeIndexingScheme {
        type BindMap = HashMap<usize, Option<usize>>;
        type Key = usize;
        type Value = usize;

        fn required_bindings(&self, key: &Self::Key) -> Vec<Self::Key> {
            if *key == 0 {
                vec![]
            } else {
                vec![key - 1]
            }
        }
    }

    impl IndexedData<TestKey> for TestData {
        type IndexingScheme = TestStrIndexingScheme;

        type Value = <Self::IndexingScheme as IndexingScheme>::Value;
        type BindMap = <Self::IndexingScheme as IndexingScheme>::BindMap;

        fn bind_options(&self, key: &TestKey, _: &Self::BindMap) -> Vec<Self::Value> {
            let key_suffix: usize = key[3..].parse().unwrap_or(0);
            vec![key_suffix]
        }
    }
    impl IndexedData<usize> for TestData {
        type IndexingScheme = TestUsizeIndexingScheme;

        fn bind_options(&self, key: &usize, known_bindings: &Self::BindMap) -> Vec<Self::Value> {
            if *key == 0 || known_bindings.get(&(key - 1)).is_some() {
                // All previous keys were assigned, we can (dummy) bind the key
                vec![*key]
            } else {
                // Require key - 1 to be bound first
                vec![]
            }
        }

        // Expose inner type aliases
        type Value = <Self::IndexingScheme as IndexingScheme>::Value;
        type BindMap = <Self::IndexingScheme as IndexingScheme>::BindMap;
    }

    impl Default for TestStrIndexingScheme {
        fn default() -> Self {
            Self
        }
    }

    #[test]
    fn test_bind_with_scheme() {
        let index_map = HashMap::default();
        let scheme = TestUsizeIndexingScheme;
        let key = 4;

        // The list of keys that must be bound before `key` can be bound
        let all_missing_keys = scheme.missing_bindings(&key, &HashSet::default());
        assert_eq!(all_missing_keys, (0..=key).collect_vec());

        // Test binding a new value
        let all_index_maps = TestData.bind_all(index_map, all_missing_keys.iter().copied());
        let index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(index_map.get_binding(&key).borrowed(), Binding::Bound(&key));

        // Must have bound all values smaller than `key` as well
        assert_eq!(index_map.len(), key + 1);

        // Test binding the same value again (should succeed)
        let all_index_maps = TestData.bind_all(index_map.clone(), all_missing_keys.iter().copied());
        let new_index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(new_index_map, index_map);

        // Test binding multiple values at the same time, with some other values
        // already preset
        let index_map = HashMap::from_iter([(3, Some(3))]);
        let all_missing_keys = scheme.all_missing_bindings([1, 4], [3]);
        assert_eq!(all_missing_keys, vec![0, 1, 4]);
        let all_index_maps = TestData.bind_all(index_map, all_missing_keys);
        let index_map = all_index_maps.into_iter().exactly_one().unwrap();
        assert_eq!(
            index_map.into_keys().sorted().collect_vec(),
            vec![0, 1, 3, 4]
        );
    }
}
