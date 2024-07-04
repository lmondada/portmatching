//! Pattern matching on 2D character grids.
//!
//! This is a straightforward generalisation of the string matcher to 2D. It
//! is interesting for testing and demonstration as overlapping but not contained
//! matches are possible.
use std::fmt::Debug;

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    indexing::BindingOptions,
    string::{StringConstraint, StringPredicate},
    IndexMap, IndexingScheme, ManyMatcher, Predicate,
};

use self::pattern::MatrixPattern;

mod pattern;
#[cfg(feature = "proptest")]
mod proptest;

/// A 2D character matrix.
pub struct MatrixString {
    /// The rows of the matrix.
    pub rows: Vec<Vec<char>>,
}

impl<S: AsRef<str>> From<S> for MatrixString {
    fn from(s: S) -> Self {
        Self {
            rows: s.as_ref().lines().map(|l| l.chars().collect()).collect(),
        }
    }
}

/// A predicate for matching a 2D character matrix.
///
/// This is the same predicate as for string matching
pub type MatrixPredicate = StringPredicate;

/// A constraint for matching a 2D character matrix.
///
/// This is the same constraint as for string matching, up to a difference in
/// indexing.
pub type MatrixConstraint = StringConstraint<MatrixIndexKey>;

/// A matcher for 2D character matrices.
pub type MatrixManyMatcher =
    ManyMatcher<MatrixPattern, MatrixIndexKey, MatrixPredicate, MatrixIndexingScheme>;

impl Predicate<MatrixString> for MatrixPredicate {
    type Value = MatrixPosition;

    fn check(&self, data: &MatrixString, args: &[&MatrixPosition]) -> bool {
        match self {
            MatrixPredicate::BindingEq => {
                let (MatrixPosition(row1, col1), MatrixPosition(row2, col2)) =
                    args.iter().collect_tuple().unwrap();
                let char1 = data.rows.get(*row1).and_then(|row| row.get(*col1));
                let char2 = data.rows.get(*row2).and_then(|row| row.get(*col2));
                char1.is_some() && char1 == char2
            }
            MatrixPredicate::ConstVal(c) => {
                let MatrixPosition(row, col) = args.iter().exactly_one().unwrap();
                data.rows.get(*row).and_then(|row| row.get(*col)) == Some(c)
            }
        }
    }
}

/// Simple indexing schemes for matrices.
///
/// The indexing scheme is a pair of (row, col) positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixIndexingScheme;

impl IndexingScheme<MatrixString, MatrixPosition> for MatrixIndexingScheme {
    type Key = MatrixIndexKey;

    fn valid_bindings(
        &self,
        key: &Self::Key,
        known_bindings: &impl IndexMap<Self::Key, MatrixPosition>,
        data: &MatrixString,
    ) -> BindingOptions<Self::Key, MatrixPosition> {
        if key == &Self::Key::root() {
            // For the root binding, any matrix position is valid
            BindingOptions::ValidBindings(
                (0..data.rows.len())
                    .flat_map(|row| (0..data.rows[row].len()).map(move |col| (row, col)))
                    .map_into()
                    .collect(),
            )
        } else {
            let Some(MatrixPosition(root_row, root_col)) = known_bindings.get(&Self::Key::root())
            else {
                return BindingOptions::MissingIndexKeys(vec![Self::Key::root()]);
            };

            let &MatrixIndexKey(offset_row, offset_col) = key;

            BindingOptions::ValidBindings(vec![
                (root_row + offset_row, root_col + offset_col).into()
            ])
        }
    }
}

/// A position in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct MatrixPosition(usize, usize);

/// An index key, obtained from the position index of the character in the pattern.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct MatrixIndexKey(usize, usize);

impl MatrixIndexKey {
    fn root() -> Self {
        Self(0, 0)
    }
}

impl Debug for MatrixIndexKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "char@{:?}", (self.0, self.1))
    }
}
