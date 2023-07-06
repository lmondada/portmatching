use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(crate) struct EmptyOr<I> {
    iter: Option<I>,
}

impl<I> EmptyOr<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self { iter: Some(iter) }
    }

    pub(crate) fn empty() -> Self {
        Self { iter: None }
    }
}

impl<I: Iterator> Iterator for EmptyOr<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter?.next()
    }
}
