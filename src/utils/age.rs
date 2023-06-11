use std::collections::BTreeSet;

pub trait Age {
    fn merge(&mut self, other: &Self);
    fn next(&self) -> Self;
}

impl Age for usize {
    fn merge(&mut self, other: &Self) {
        if self != other {
            panic!("Cannot merge ages {:?} and {:?}", self, other);
        }
    }

    fn next(&self) -> Self {
        *self + 1
    }
}

impl Age for BTreeSet<usize> {
    fn merge(&mut self, other: &Self) {
        self.extend(other.iter().copied());
    }

    fn next(&self) -> Self {
        if self.len() != 1 {
            panic!("Cannot increment age {:?}", self);
        }
        [self.first().unwrap() + 1].into()
    }
}
