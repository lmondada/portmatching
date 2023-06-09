pub trait Age {
    fn merge(&self, other: &Self) -> Self;
    fn next(&self) -> Self;
}

impl Age for usize {
    fn merge(&self, other: &Self) -> Self {
        if self != other {
            panic!("Cannot merge ages {:?} and {:?}", self, other);
        }
        *self
    }

    fn next(&self) -> Self {
        *self + 1
    }
}

impl Age for Vec<usize> {
    fn merge(&self, other: &Self) -> Self {
        self.iter().chain(other.iter()).copied().collect()
    }

    fn next(&self) -> Self {
        if self.len() != 1 {
            panic!("Cannot increment age {:?}", self);
        }
        vec![self[0] + 1]
    }
}
