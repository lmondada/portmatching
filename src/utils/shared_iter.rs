use std::{cell::RefCell, rc::Rc};

#[derive(Clone)]
pub(crate) struct SharedIter<I> {
    iter: Rc<RefCell<I>>,
}

impl<I> SharedIter<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self {
            iter: Rc::new(RefCell::new(iter)),
        }
    }
}

impl<I: Iterator> Iterator for SharedIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.borrow_mut().next()
    }
}
