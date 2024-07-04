use std::marker::PhantomData;

use crate::{utils::UniqueStack, HashSet, IndexMap};

use super::{BindVariableError, IndexKey};

pub(super) struct BindingBuilder<K, V, M> {
    bindings: M,
    /// A stack of the missing keys.
    ///
    /// It's important that we process the keys first-in last-out, to ensure
    /// that we process the dependencies we added to the stack before processing
    /// the same key again.
    missing_keys: UniqueStack<K>,
    exclude_keys: HashSet<K>,
    _phantom: PhantomData<V>,
}

impl<K: Clone, V, M: Clone> Clone for BindingBuilder<K, V, M> {
    fn clone(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            missing_keys: self.missing_keys.clone(),
            exclude_keys: self.exclude_keys.clone(),
            _phantom: PhantomData::<V>,
        }
    }
}

impl<K: IndexKey, V, M: IndexMap<K, V>> BindingBuilder<K, V, M> {
    pub(super) fn new(keys: impl IntoIterator<Item = K>, bindings: M) -> Self {
        let missing_keys = keys.into_iter().collect();
        Self {
            missing_keys,
            bindings,
            exclude_keys: HashSet::default(),
            _phantom: PhantomData::<V>,
        }
    }

    pub(super) fn bindings(&self) -> &M {
        &self.bindings
    }

    pub(super) fn top_missing_key(&mut self) -> Option<K> {
        let mut key = *self.missing_keys.top()?;
        while self.bindings.get(&key).is_some() || self.exclude_keys.contains(&key) {
            // Already bound or excluded
            self.missing_keys.pop();
            key = *self.missing_keys.top()?;
        }
        Some(key)
    }

    pub(super) fn finish(self) -> M {
        self.bindings
    }

    pub(super) fn apply_bindings(
        mut self,
        key: K,
        values: impl IntoIterator<Item = V>,
    ) -> Result<Vec<Self>, BindVariableError> {
        if self.missing_keys.top() == Some(&key) {
            self.missing_keys.pop();
        }
        values
            .into_iter()
            .map(|value| {
                let mut new_self = self.clone();
                new_self.bindings.bind(key, value)?;
                Ok(new_self)
            })
            .collect()
    }

    pub(super) fn exclude_key(&mut self, key: K) {
        if self.missing_keys.top() == Some(&key) {
            self.missing_keys.pop();
        }
        self.exclude_keys.insert(key);
    }

    /// Extend the missing key set.
    ///
    /// Return whether any key was added.
    pub(super) fn extend_missing_keys(&mut self, keys: impl IntoIterator<Item = K>) -> bool {
        let mut to_add = keys
            .into_iter()
            .filter(|key| !self.exclude_keys.contains(key))
            .peekable();
        if to_add.peek().is_some() {
            self.missing_keys.extend(to_add);
            true
        } else {
            false
        }
    }
}
