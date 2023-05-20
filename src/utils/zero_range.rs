use std::{cmp, ops::RangeInclusive};

/// A range around zero
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) enum ZeroRange {
    NonEmpty {
        // the negative of the start
        start: usize,
        end: usize,
    },
    Empty,
}

impl ZeroRange {
    pub(crate) fn contains(&self, n: isize) -> bool {
        n >= self.start() && n <= self.end()
    }

    pub(crate) fn start(&self) -> isize {
        match self {
            &ZeroRange::NonEmpty { start, .. } => -(start as isize),
            ZeroRange::Empty => 0,
        }
    }

    pub(crate) fn end(&self) -> isize {
        match self {
            &ZeroRange::NonEmpty { end, .. } => end as isize,
            ZeroRange::Empty => -1,
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        let &Self::NonEmpty { start: o_start, end: o_end } = other else {
            return;
        };
        self.insert(-(o_start as isize));
        self.insert(o_end as isize)
    }

    pub(crate) fn insert(&mut self, val: isize) {
        match self {
            Self::NonEmpty { start, end } => {
                if val < 0 {
                    *start = cmp::max(*start, -val as usize);
                } else {
                    *end = cmp::max(*end, val as usize);
                }
            }
            Self::Empty => {
                *self = Self::NonEmpty {
                    start: if val < 0 { -val as usize } else { 0 },
                    end: if val < 0 { 0 } else { val as usize },
                }
            }
        }
    }
}

impl TryFrom<RangeInclusive<isize>> for ZeroRange {
    type Error = ();

    fn try_from(value: RangeInclusive<isize>) -> Result<Self, Self::Error> {
        let start = if *value.start() > 0 {
            return Err(());
        } else {
            -value.start() as usize
        };
        let end = if *value.end() < 0 {
            return Err(());
        } else {
            *value.end() as usize
        };
        Ok(ZeroRange::NonEmpty { start, end })
    }
}
