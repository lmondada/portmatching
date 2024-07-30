use std::fmt;

use crate::{utils::UniqueStack, HashSet, IndexMap};

use super::BindVariableError;

pub(super) struct BindingBuilder<M: IndexMap> {
    bindings: M,
    /// A stack of the missing keys.
    ///
    /// It's important that we process the keys first-in last-out, to ensure
    /// that we process the dependencies we added to the stack before processing
    /// the same key again.
    missing_keys: UniqueStack<M::Key>,
    exclude_keys: HashSet<M::Key>,
}

impl<M: IndexMap> Clone for BindingBuilder<M> {
    fn clone(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            missing_keys: self.missing_keys.clone(),
            exclude_keys: self.exclude_keys.clone(),
        }
    }
}

impl<M: IndexMap + fmt::Debug> fmt::Debug for BindingBuilder<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BindingBuilder{:?}", self.bindings)
    }
}

impl<M: IndexMap> BindingBuilder<M> {
    pub fn new(keys: impl IntoIterator<Item = M::Key>, bindings: M) -> Self {
        let missing_keys = keys.into_iter().collect();
        Self {
            missing_keys,
            bindings,
            exclude_keys: HashSet::default(),
        }
    }

    pub fn bindings(&self) -> &M {
        &self.bindings
    }

    pub fn top_missing_key(&mut self) -> Option<M::Key> {
        let mut key = *self.missing_keys.top()?;
        while self.bindings.get(&key).is_some() || self.exclude_keys.contains(&key) {
            // Already bound or excluded
            self.missing_keys.pop();
            key = *self.missing_keys.top()?;
        }
        Some(key)
    }

    pub fn finish(self) -> M {
        self.bindings
    }

    pub fn apply_bindings(
        mut self,
        key: M::Key,
        values: impl IntoIterator<Item = M::Value>,
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

    pub fn exclude_key(&mut self, key: M::Key) {
        if self.missing_keys.top() == Some(&key) {
            self.missing_keys.pop();
        }
        self.exclude_keys.insert(key);
    }

    /// Extend the missing key set.
    ///
    /// Return whether any key was added.
    pub fn extend_missing_keys(&mut self, keys: impl IntoIterator<Item = M::Key>) -> bool {
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
