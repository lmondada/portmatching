use std::iter::{FusedIterator, Peekable};

/// Mark the last element of an iterator as true and return the rest as false.
pub(crate) fn mark_last<T>(iter: impl IntoIterator<Item = T>) -> impl Iterator<Item = (T, bool)> {
    MarkLast::new(iter)
}

pub struct MarkLast<I: Iterator> {
    iter: Peekable<I>,
}

impl<I: Iterator> MarkLast<I> {
    fn new(iter: impl IntoIterator<IntoIter = I>) -> Self {
        Self {
            iter: iter.into_iter().peekable(),
        }
    }
}

impl<I: Iterator> Iterator for MarkLast<I> {
    type Item = (I::Item, bool);

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next()?;
        let is_last = self.iter.peek().is_none();
        Some((next, is_last))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for MarkLast<I> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<I: FusedIterator> FusedIterator for MarkLast<I> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_last() {
        let iter = 0..5;
        let result = mark_last(iter).collect::<Vec<_>>();
        assert_eq!(
            result,
            [(0, false), (1, false), (2, false), (3, false), (4, true)]
        );
    }
}
