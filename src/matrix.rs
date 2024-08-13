//! Pattern matching on 2D character grids.
//!
//! This is a straightforward generalisation of the string matcher to 2D. It
//! is interesting for testing and demonstration as overlapping but not contained
//! matches are possible.
use std::{
    borrow::Borrow,
    fmt::{self, Debug},
};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    indexing::{BindVariableError, BindingResult, MissingIndexKeys, ValidBindings},
    string::{CharacterPredicate, StringConstraint},
    IndexMap, IndexingScheme, ManyMatcher, Predicate,
};

pub use self::pattern::MatrixPattern;

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

/// A constraint for matching a 2D character matrix.
///
/// This is the same constraint as for string matching, up to a difference in
/// indexing.
pub type MatrixConstraint = StringConstraint<MatrixPatternPosition>;

/// A matcher for 2D character matrices.
pub type MatrixManyMatcher =
    ManyMatcher<MatrixPattern, MatrixPatternPosition, CharacterPredicate, MatrixIndexingScheme>;

impl Predicate<MatrixString> for CharacterPredicate {
    type Value = MatrixSubjectPosition;

    fn check(&self, data: &MatrixString, args: &[impl Borrow<MatrixSubjectPosition>]) -> bool {
        match self {
            CharacterPredicate::BindingEq => {
                let (MatrixSubjectPosition(row1, col1), MatrixSubjectPosition(row2, col2)) =
                    args.iter().map(|pos| pos.borrow()).collect_tuple().unwrap();
                let char1 = data.rows.get(*row1).and_then(|row| row.get(*col1));
                let char2 = data.rows.get(*row2).and_then(|row| row.get(*col2));
                char1.is_some() && char1 == char2
            }
            CharacterPredicate::ConstVal(c) => {
                let MatrixSubjectPosition(row, col) = args
                    .iter()
                    .map(|pos| pos.borrow())
                    .exactly_one()
                    .ok()
                    .unwrap();
                data.rows.get(*row).and_then(|row| row.get(*col)) == Some(c)
            }
        }
    }
}

/// A map for matrix positions.
///
/// Only store the position of the start of the match, compute other positions by
/// adding row & column offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum MatrixPositionMap {
    /// No key has been bound yet
    #[default]
    Unbound,
    /// Keys in the submatrix between `min_pos` and `max_pos` are valid and
    /// computed as offset from the start position
    Bound {
        /// The position of the start of the pattern in the string
        start_pos: MatrixSubjectPosition,
        /// The interval matched so far (start)
        min_pos: MatrixPatternPosition,
        /// The interval matched so far (end)
        max_pos: MatrixPatternPosition,
    },
}

impl MatrixPositionMap {
    fn start_pos(&self) -> Option<MatrixSubjectPosition> {
        match self {
            Self::Unbound => None,
            &Self::Bound { start_pos, .. } => Some(start_pos),
        }
    }
}

impl IndexMap for MatrixPositionMap {
    type Key = MatrixPatternPosition;
    type Value = MatrixSubjectPosition;
    type ValueRef<'a> = MatrixSubjectPosition;

    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>> {
        let Self::Bound {
            start_pos,
            min_pos,
            max_pos,
        } = self
        else {
            return None;
        };
        let &MatrixPatternPosition(key_row, key_col) = var;
        if key_row >= min_pos.0
            && key_row < max_pos.0
            && key_col >= min_pos.1
            && key_col < max_pos.1
        {
            Some(MatrixSubjectPosition(
                start_pos.0.checked_add_signed(key_row).unwrap(),
                start_pos.1.checked_add_signed(key_col).unwrap(),
            ))
        } else {
            None
        }
    }

    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), BindVariableError> {
        let MatrixPatternPosition(key_row, key_col) = var;
        if key_row == 0 && key_col == 0 {
            if let Some(MatrixSubjectPosition(start_row, start_col)) = self.start_pos() {
                return Err(BindVariableError::VariableExists {
                    key: format!("{:?}", var),
                    curr_value: format!("({}, {})", start_row, start_col),
                    new_value: format!("{:?}", val),
                });
            } else {
                *self = Self::Bound {
                    start_pos: val,
                    min_pos: MatrixPatternPosition(0, 0),
                    max_pos: MatrixPatternPosition(1, 1),
                };
            }
        } else {
            let Self::Bound {
                min_pos, max_pos, ..
            } = self
            else {
                return Err(BindVariableError::InvalidKey {
                    key: format!("{:?}", var),
                });
            };
            *min_pos = min_pos.cwise_min(&var);
            *max_pos = max_pos.cwise_max(&var);
        }
        Ok(())
    }
}

/// Simple indexing schemes for matrices.
///
/// The indexing scheme is a pair of (row, col) positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixIndexingScheme;

impl IndexingScheme<MatrixString> for MatrixIndexingScheme {
    type Map = MatrixPositionMap;

    fn valid_bindings(
        &self,
        key: &MatrixPatternPosition,
        known_bindings: &MatrixPositionMap,
        data: &MatrixString,
    ) -> BindingResult<Self, MatrixString> {
        let &MatrixPatternPosition(key_row, key_col) = key;

        if key_row == 0 && key_col == 0 {
            // No key has been bound yet
            assert!(matches!(known_bindings, Self::Map::Unbound));
            // For the root binding, any matrix position is valid
            Ok(ValidBindings(
                (0..data.rows.len())
                    .flat_map(|row| (0..data.rows[row].len()).map(move |col| (row, col)))
                    .map_into()
                    .collect(),
            ))
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            let Some(MatrixSubjectPosition(start_row, start_col)) = known_bindings.start_pos()
            else {
                return Err(MissingIndexKeys(vec![MatrixPatternPosition::start()]));
            };

            Ok(ValidBindings(vec![(
                start_row.checked_add_signed(key_row).unwrap(),
                start_col.checked_add_signed(key_col).unwrap(),
            )
                .into()]))
        }
    }
}

/// A position in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct MatrixSubjectPosition(usize, usize);

/// An index key, obtained from the position index of the character in the pattern.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct MatrixPatternPosition(isize, isize);

impl MatrixPatternPosition {
    /// The root index key.
    pub fn start() -> Self {
        Self(0, 0)
    }

    fn cwise_max(&self, other: &Self) -> Self {
        Self(self.0.max(other.0), self.1.max(other.1))
    }

    fn cwise_min(&self, other: &Self) -> Self {
        Self(self.0.min(other.0), self.1.min(other.1))
    }
}

impl Debug for MatrixPatternPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "char@{:?}", (self.0, self.1))
    }
}

impl Debug for MatrixString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.rows {
            for c in row {
                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{string::tests::define_matcher_factories, PatternMatch, PortMatcher};

    use super::*;
    use rstest::rstest;

    pub(super) fn apply_all_matchers(
        patterns: Vec<MatrixPattern>,
        subject: &MatrixString,
    ) -> [Vec<PatternMatch<MatrixPositionMap>>; 2] {
        const MATCHER_FACTORIES: &[fn(Vec<MatrixPattern>) -> MatrixManyMatcher] =
            define_matcher_factories!(MatrixPattern, MatrixManyMatcher);

        // Skip the all deterministic matcher, too slow
        let all_matches = MATCHER_FACTORIES[..2]
            .iter()
            .map(|matcher_factory| {
                matcher_factory(patterns.clone())
                    .find_matches(subject)
                    .collect_vec()
            })
            .collect_vec();
        all_matches.into_iter().collect_vec().try_into().unwrap()
    }

    #[rstest]
    #[case("f\n", vec!["f\n", "", "f\n"])]
    fn proptest_fail_cases(#[case] subject: &str, #[case] patterns: Vec<&str>) {
        let subject = MatrixString::from(&subject);
        let patterns = patterns
            .into_iter()
            .map(MatrixPattern::parse_str)
            .collect_vec();
        let [mut non_det, mut default] = apply_all_matchers(patterns, &subject);

        // dbg!(&all_matches);
        // println!(
        //     "{:?}",
        //     all_matches
        //         .iter()
        //         .map(|m| m.iter().map(|m| m.pattern).collect_vec())
        //         .collect_vec()
        // );
        non_det.sort();
        default.sort();
        assert_eq!(non_det, default);
    }
}
