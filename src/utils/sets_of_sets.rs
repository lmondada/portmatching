use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
};

/// A simplified union find
pub(crate) struct SetsOfSets<T> {
    sets: BTreeMap<T, BTreeSet<T>>,
    sets_inv: BTreeMap<T, T>,
}

impl<T> SetsOfSets<T> {
    pub(crate) fn new() -> Self {
        Self {
            sets: BTreeMap::new(),
            sets_inv: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, x: &T, y: T)
    where
        T: Ord + Clone,
    {
        if !self.sets_inv.contains_key(x) {
            self.singleton(x);
        }
        let repr = &self.sets_inv[x];
        self.sets
            .get_mut(repr)
            .expect("always present")
            .insert(y.clone());
        self.sets_inv.insert(y, repr.clone());
    }

    pub(crate) fn get(&mut self, x: &T) -> &BTreeSet<T>
    where
        T: Ord + Clone,
    {
        if !self.sets_inv.contains_key(x) {
            self.singleton(x);
        }
        let repr = &self.sets_inv[x];
        &self.sets[repr]
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut BTreeSet<T>> {
        self.sets.values_mut()
    }

    fn singleton(&mut self, x: &T)
    where
        T: Ord + Clone,
    {
        self.sets_inv.insert(x.clone(), x.clone());
        self.sets
            .insert(x.clone(), BTreeSet::from_iter([x.clone()]));
    }
}

impl<T: Debug> Debug for SetsOfSets<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for set in self.sets.values() {
            writeln!(f, "{:?}", set)?
        }
        Ok(())
    }
}
