use std::cmp::{Ordering, Reverse};

/// A trait for iterators to collect all minimum values (according to some key).
pub(crate) trait CollectMin<K>: Iterator {
    /// Filters and collects all items with the minimum key.
    ///
    /// Return the collected elements and the minimum key.
    fn collect_min_by_key(self, key: impl Fn(&Self::Item) -> K) -> (Vec<Self::Item>, Option<K>);
}

impl<T: Iterator, K: Ord> CollectMin<K> for T {
    fn collect_min_by_key(self, key: impl Fn(&Self::Item) -> K) -> (Vec<Self::Item>, Option<K>) {
        let acc = self.fold(MinVec::new(), |mut acc, item| {
            let key = key(&item);
            acc.push(item, key);
            acc
        });
        (acc.vec, acc.min.to_val())
    }
}

#[derive(Debug, Clone)]
struct MinVec<V, K> {
    vec: Vec<V>,
    min: ValOrInf<K>,
}

impl<V, K> MinVec<V, K> {
    fn new() -> Self {
        Self {
            vec: vec![],
            min: ValOrInf::inf(),
        }
    }

    fn push(&mut self, item: V, key: K)
    where
        K: Ord,
    {
        match ValOrInf::from_val(&key).cmp(&self.min.as_ref()) {
            Ordering::Less => {
                self.min = ValOrInf::from_val(key);
                self.vec.clear();
                self.vec.push(item);
            }
            Ordering::Equal => {
                self.vec.push(item);
            }
            Ordering::Greater => {}
        }
    }
}

impl<V, K> Default for MinVec<V, K> {
    fn default() -> Self {
        Self::new()
    }
}

/// An Option<V> type but where None is the largest value
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ValOrInf<V>(Reverse<Option<Reverse<V>>>);

impl<V> ValOrInf<V> {
    pub fn inf() -> Self {
        ValOrInf(Reverse(None))
    }

    pub fn from_val(val: V) -> Self {
        ValOrInf(Reverse(Some(Reverse(val))))
    }

    pub fn to_val(self) -> Option<V> {
        self.0 .0.map(|x| x.0)
    }

    pub fn as_ref(&self) -> ValOrInf<&V> {
        match self.0 .0.as_ref() {
            Some(val) => ValOrInf::from_val(&val.0),
            None => ValOrInf::inf(),
        }
    }
}

impl<V: PartialOrd> PartialOrd for ValOrInf<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<V: Ord> Ord for ValOrInf<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<V> From<Option<V>> for ValOrInf<V> {
    fn from(val: Option<V>) -> Self {
        match val {
            Some(val) => ValOrInf::from_val(val),
            None => ValOrInf::inf(),
        }
    }
}

impl<V> From<ValOrInf<V>> for Option<V> {
    fn from(val: ValOrInf<V>) -> Self {
        val.to_val()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_vec() {
        let mut vec = MinVec::new();
        vec.push(1, 1);
        assert_eq!(vec.min, ValOrInf::from_val(1));
        vec.push(2, 2);
        assert_eq!(vec.min, ValOrInf::from_val(1));
        vec.push(3, 3);
        assert_eq!(vec.min, ValOrInf::from_val(1));
        vec.push(2, 1);
        assert_eq!(vec.min, ValOrInf::from_val(1));
        assert_eq!(vec.vec, vec![1, 2]);
        vec.push(2, 0);
        assert_eq!(vec.min, ValOrInf::from_val(0));
        assert_eq!(vec.vec, vec![2]);
    }

    #[test]
    fn test_collect_min_by_key() {
        let (vec, min) = (1..=6).collect_min_by_key(|x| x % 2);
        assert_eq!(vec, vec![2, 4, 6]);
        assert_eq!(min, Some(0));
    }
}
