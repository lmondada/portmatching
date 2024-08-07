//! A stack that only allows unique elements.

use std::{fmt::Debug, hash::Hash, iter};

use itertools::Itertools;

use crate::HashMap;

/// An element of the stack
#[derive(Clone, Debug)]
struct El<T> {
    /// The value of the element
    value: T,
    /// The index of the next element in the stack
    below: Option<usize>,
    /// The index of the previous element in the stack
    above: Option<usize>,
}

/// Stack with unique elements.
///
/// A first in first out stack. If pushing an element that is already
/// in the stack, the operation moves that element to the top of the stack.
//
// TODO: instead of `free_indices`, one could use below and above to create
// a second linked list of free indices.
#[derive(Clone)]
pub(crate) struct UniqueStack<T> {
    elements: Vec<Option<El<T>>>,
    indices: HashMap<T, usize>,
    top: Option<usize>,
    free_indices: Vec<usize>,
}

impl<T: Eq + Hash + Clone> UniqueStack<T> {
    /// Create a new empty UniqueStack.
    pub fn new() -> Self {
        UniqueStack {
            elements: Vec::new(),
            indices: HashMap::default(),
            top: None,
            free_indices: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        UniqueStack {
            elements: Vec::with_capacity(capacity),
            ..Self::new()
        }
    }

    /// Push an element onto the stack, moving it to the top if already present.
    ///
    /// Return true if the element was added, false if it was already present.
    pub fn push(&mut self, value: T) -> bool {
        if let Some(&index) = self.indices.get(&value) {
            self.rewire_top(index);
            false
        } else {
            // New element, add it to the top
            let new_index = self.get_free_index();
            let el = El {
                value: value.clone(),
                below: None,
                above: None,
            };
            self.elements[new_index] = Some(el);
            self.indices.insert(value, new_index);
            self.rewire_top(new_index);
            true
        }
    }

    /// Remove and return the top element from the stack.
    ///
    /// Return None if the stack is empty.
    pub fn pop(&mut self) -> Option<T> {
        let top_index = self.top?;
        let el = self.elements[top_index].take().unwrap();
        self.indices.remove(&el.value);
        self.free_indices.push(top_index);
        self.top = el.below;
        if let Some(new_top) = self.top {
            self.elements[new_top].as_mut().unwrap().above = None;
        }
        Some(el.value)
    }

    /// The top element of the stack without removing it.
    ///
    /// Return None if the stack is empty.
    pub fn top(&self) -> Option<&T> {
        self.top
            .map(|index| &self.elements[index].as_ref().unwrap().value)
    }

    /// Iterate over the elements in the stack
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        let mut next_index = self.top;
        iter::from_fn(move || {
            let index = next_index.take()?;
            let el = self.elements[index].as_ref().unwrap();
            next_index = el.below;
            Some(&el.value)
        })
    }

    /// A free spot in the vector, creating it if necessary
    fn get_free_index(&mut self) -> usize {
        self.free_indices.pop().unwrap_or_else(|| {
            self.elements.push(None);
            self.elements.len() - 1
        })
    }

    /// Place index at top of the stack
    fn rewire_top(&mut self, index: usize) {
        let prev_top = self.top.replace(index);
        if Some(index) == prev_top {
            return;
        }
        let &El { below, above, .. } = self.elements[index].as_ref().unwrap();
        self.link(below, above);
        self.link(prev_top, Some(index));
        self.elements[index].as_mut().unwrap().above = None;
    }

    /// Link two elements together by setting `above` and `below` indices
    fn link(&mut self, below: Option<usize>, above: Option<usize>) {
        if let Some(below) = below {
            self.elements[below].as_mut().unwrap().above = above;
        }
        if let Some(above) = above {
            self.elements[above].as_mut().unwrap().below = below;
        }
    }
}

impl<T: Eq + Hash + Clone> FromIterator<T> for UniqueStack<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // TODO: rewrite to skip some redundant checks here
        let iter = iter.into_iter();
        let mut stack = UniqueStack::with_capacity(iter.size_hint().0);
        for item in iter {
            stack.push(item);
        }
        stack
    }
}

impl<T: Eq + Hash + Clone> Extend<T> for UniqueStack<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Eq + Hash + Clone + Debug> Debug for UniqueStack<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut stack = self.iter().collect_vec();
        stack.reverse();
        write!(f, "UniqueStack{:?}", stack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_pop() {
        let mut stack = UniqueStack::new();
        assert!(stack.push(1));
        assert!(stack.push(2));
        assert!(stack.push(3));
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_push_existing() {
        let mut stack = UniqueStack::new();
        assert!(stack.push(1));
        assert!(stack.push(2));
        assert!(!stack.push(1)); // Should return false for existing element
        assert_eq!(stack.pop(), Some(1)); // 1 should now be on top
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_top() {
        let mut stack = UniqueStack::new();
        assert_eq!(stack.top(), None);
        stack.push(1);
        assert_eq!(stack.top(), Some(&1));
        stack.push(2);
        assert_eq!(stack.top(), Some(&2));
    }

    use rstest::rstest;

    #[rstest]
    #[case(vec![1, 2, 3, 2, 1], vec![1, 2, 3])]
    #[case(vec![0, 0], vec![0])]
    fn test_from_iter(#[case] input: Vec<i32>, #[case] expected: Vec<i32>) {
        let mut stack: UniqueStack<i32> = input.into_iter().collect();

        for &expected_value in expected.iter() {
            assert_eq!(stack.pop(), Some(expected_value));
        }
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_extend() {
        let mut stack = UniqueStack::new();
        stack.push(1);
        stack.extend(vec![2, 3, 1, 4]);
        assert_eq!(stack.pop(), Some(4));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), None);
    }

    #[test]
    fn test_iter_debug() {
        let vec = vec![1, 2, 3];
        let stack: UniqueStack<i32> = vec.iter().copied().collect();
        assert_eq!(format!("{:?}", stack), format!("UniqueStack{:?}", vec));
    }
}

#[cfg(test)]
#[cfg(feature = "proptest")]
mod proptests {
    use std::iter;

    use itertools::Itertools;
    use proptest::prelude::*;

    use super::*;
    use crate::HashSet;

    proptest! {
        #[test]
        fn proptest_unique_stack(input in prop::collection::vec(0..6usize, 0..50)) {
            let mut stack: UniqueStack<usize> = input.iter().cloned().collect();

            // Create the expected result by traversing the input vector from back to front
            let mut expected = Vec::new();
            let mut seen = HashSet::default();
            for &item in input.iter().rev() {
                if seen.insert(item) {
                    expected.push(item);
                }
            }

            // Drain the stack and compare with expected result
            let actual = iter::from_fn(|| stack.pop()).collect_vec();

            assert_eq!(actual, expected, "Stack contents do not match expected unique elements");
        }
    }
}
