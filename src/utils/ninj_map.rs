use std::{
    collections::{btree_map::IntoIter, BTreeMap, BTreeSet},
    fmt::Debug,
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NInjMap<L, R> {
    left: BTreeMap<L, R>,
    right: BTreeMap<R, BTreeSet<L>>,
}

impl<L, R> IntoIterator for NInjMap<L, R> {
    type Item = (L, R);

    type IntoIter = IntoIter<L, R>;

    fn into_iter(self) -> Self::IntoIter {
        self.left.into_iter()
    }
}

impl<L: Debug, R: Debug> Debug for NInjMap<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (l, r) in self.iter() {
            writeln!(f, "{l:?} <> {r:?}")?;
        }
        Ok(())
    }
}

impl<L, R> Default for NInjMap<L, R> {
    fn default() -> Self {
        Self {
            left: Default::default(),
            right: Default::default(),
        }
    }
}

impl<L: Ord + Clone, R: Ord + Clone> FromIterator<(L, R)> for NInjMap<L, R> {
    fn from_iter<T: IntoIterator<Item = (L, R)>>(iter: T) -> Self {
        let mut ret = NInjMap::new();
        for (l, r) in iter {
            ret.insert(l, r);
        }
        ret
    }
}

impl<L, R> NInjMap<L, R> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&L, &R)> {
        self.left.iter()
    }
}

impl<L: Ord + Clone, R: Ord + Clone> NInjMap<L, R> {
    pub fn get_by_left(&self, left_key: &L) -> Option<&R> {
        self.left.get(left_key)
    }

    pub fn get_by_right_iter(&self, right_key: &R) -> impl Iterator<Item = &L> {
        self.right
            .get(right_key)
            .into_iter()
            .flat_map(|set| set.iter())
    }

    pub fn get_by_right(&self, right_key: &R) -> Option<&BTreeSet<L>> {
        self.right.get(right_key)
    }

    pub fn contains_right(&self, right_key: &R) -> bool {
        self.right.contains_key(right_key)
    }

    /// Insert pair (l, r) in map
    ///
    /// Returns whether the insertion was successful, i.e. if the key
    /// `left` was not already present
    pub fn insert(&mut self, left: L, right: R) -> bool {
        if self.left.insert(left.clone(), right.clone()).is_some() {
            return false;
        }
        self.right.entry(right).or_default().insert(left);
        true
    }

    pub fn intersect(&mut self, other: &Self) {
        for (l, r) in other.iter() {
            self.insert(l.clone(), r.clone());
        }
    }
}
