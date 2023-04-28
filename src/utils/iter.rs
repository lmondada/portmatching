use crate::addressing::SpineAddress;

#[derive(Debug, Clone)]
pub struct AddAsRefIt<I> {
    iter: I,
}

impl<I> AddAsRefIt<I> {
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<'a, A: SpineAddress + 'a, I: Iterator<Item = &'a A>> Iterator for AddAsRefIt<I> {
    type Item = A::AsRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|addr| addr.as_ref())
    }
}

// #[derive(Debug, Clone)]
// pub struct CopySliceCopyIt<I> {
//     iter: I,
// }

// impl<I> CopySliceCopyIt<I> {
//     pub fn new(iter: I) -> Self {
//         Self { iter }
//     }
// }

// impl<'a, A: 'a + Copy, B: 'a, C: 'a + Copy, I: Iterator<Item = &'a (A, Vec<B>, C)>> Iterator for CopySliceCopyIt<I> {
//     type Item = (A, &'a [B], C);

//     fn next(&mut self) -> Option<Self::Item> {
//         self.iter.next().map(|(a, b, c)| (*a, b.as_slice(), *c))
//     }
// }