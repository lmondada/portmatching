//! Pattern matching on 2D character grids.
//!
//! This is a straightforward generalisation of the string matcher to 2D. It
//! is interesting for testing and demonstration as overlapping but not contained
//! matches are possible.
use std::{
    borrow::Borrow,
    collections::BTreeSet,
    fmt::{self, Debug},
};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    concrete::string::CharacterPredicate,
    indexing::{BindVariableError, Binding, IndexedData},
    predicate::{DeterministicPredicatePatternSelector, PredicatePatternDefaultSelector},
    BindMap, IndexingScheme, ManyMatcher, NaiveManyMatcher, Predicate,
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
pub type MatrixConstraint = crate::Constraint<MatrixPatternPosition, CharacterPredicate>;

type BranchSelector =
    DeterministicPredicatePatternSelector<MatrixPatternPosition, CharacterPredicate>;

/// A matcher for 2D character matrices.
pub type MatrixManyMatcher = ManyMatcher<MatrixPattern, MatrixPatternPosition, BranchSelector>;
/// A naive matcher for 2D character matrices.
///
/// Only use for testing, as it is too slow for practical purposes.
pub type MatrixNaiveManyMatcher = NaiveManyMatcher<MatrixPatternPosition, BranchSelector>;

impl Predicate<MatrixString, MatrixSubjectPosition> for CharacterPredicate {
    fn check(&self, args: &[impl Borrow<MatrixSubjectPosition>], data: &MatrixString) -> bool {
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct MatrixPositionMap {
    /// The position of the start of the pattern in the string
    ///
    /// None if no variable has been bound yet.
    start_pos: Option<MatrixSubjectPosition>,
    /// The interval matched so far, inclusive (start)
    ///
    /// If `start_pos` is `None`, then this is (0, 0) and is meaningless.
    min_pos: MatrixPatternPosition,
    /// The interval matched so far, inclusive (end)
    ///
    /// If `start_pos` is `None`, then this is (0, 0) and is meaningless.
    max_pos: MatrixPatternPosition,

    /// A set of positions that we have failed to bind
    failed_bindings: BTreeSet<MatrixPatternPosition>,
}

impl MatrixPositionMap {
    fn start_pos(&self) -> Option<MatrixSubjectPosition> {
        self.start_pos
    }

    fn is_unbound(&self) -> bool {
        self.start_pos.is_none()
    }
}

impl BindMap for MatrixPositionMap {
    type Key = MatrixPatternPosition;
    type Value = MatrixSubjectPosition;

    fn get_binding(&self, var: &Self::Key) -> Binding<impl Borrow<Self::Value> + '_> {
        let Self {
            start_pos,
            min_pos,
            max_pos,
            failed_bindings,
        } = self;
        if failed_bindings.contains(var) {
            return Binding::Failed;
        }
        let Some(start_pos) = start_pos else {
            return Binding::Unbound;
        };
        let &MatrixPatternPosition(key_row, key_col) = var;
        if key_row < min_pos.0 || key_row > max_pos.0 || key_col < min_pos.1 || key_col > max_pos.1
        {
            return Binding::Unbound;
        }
        Binding::Bound(MatrixSubjectPosition(
            start_pos.0.checked_add_signed(key_row).unwrap(),
            start_pos.1.checked_add_signed(key_col).unwrap(),
        ))
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
                self.start_pos = Some(val);
            }
        } else if self.is_unbound() {
            return Err(BindVariableError::InvalidKey {
                key: format!("{:?}", var),
            });
        }

        let Self {
            min_pos, max_pos, ..
        } = self;

        *min_pos = min_pos.cwise_min(&var);
        *max_pos = max_pos.cwise_max(&var);
        Ok(())
    }

    fn bind_failed(&mut self, var: Self::Key) {
        let Self {
            failed_bindings, ..
        } = self;
        failed_bindings.insert(var);
    }
}

/// Simple indexing schemes for matrices.
///
/// The indexing scheme is a pair of (row, col) positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixIndexingScheme;

impl IndexingScheme for MatrixIndexingScheme {
    type BindMap = MatrixPositionMap;
    type Key = <Self::BindMap as BindMap>::Key;
    type Value = <Self::BindMap as BindMap>::Value;

    fn required_bindings(&self, key: &Self::Key) -> Vec<Self::Key> {
        let &MatrixPatternPosition(key_row, key_col) = key;

        if key_row == 0 && key_col == 0 {
            // The start binding does not require any previous bindings
            Vec::new()
        } else {
            // Must bind the start position first
            vec![MatrixPatternPosition::start()]
        }
    }
}

impl IndexedData<MatrixPatternPosition> for MatrixString {
    type IndexingScheme = MatrixIndexingScheme;
    type Value = <Self::IndexingScheme as IndexingScheme>::Value;
    type BindMap = <Self::IndexingScheme as IndexingScheme>::BindMap;

    fn list_bind_options(
        &self,
        key: &MatrixPatternPosition,
        known_bindings: &MatrixPositionMap,
    ) -> Vec<MatrixSubjectPosition> {
        let &MatrixPatternPosition(key_row, key_col) = key;

        if key_row == 0 && key_col == 0 {
            // No key has been bound yet
            assert!(known_bindings.is_unbound());

            // For the root binding, any matrix position is valid
            (0..self.rows.len())
                .flat_map(|row| (0..self.rows[row].len()).map(move |col| (row, col)))
                .map_into()
                .collect()
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            let Some(MatrixSubjectPosition(start_row, start_col)) = known_bindings.start_pos()
            else {
                return vec![];
            };

            let Some(new_row) = start_row.checked_add_signed(key_row) else {
                return vec![];
            };
            let Some(new_col) = start_col.checked_add_signed(key_col) else {
                return vec![];
            };
            if new_row < self.rows.len() && new_col < self.rows[new_row].len() {
                vec![(new_row, new_col).into()]
            } else {
                vec![]
            }
        }
    }
}

/// A position in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct MatrixSubjectPosition(usize, usize);

/// An index key, obtained from the position index of the character in the pattern.
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
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
    use crate::{
        concrete::string::tests::{define_matcher_factories, Matcher},
        PatternID, PatternMatch, PortMatcher,
    };

    use super::*;
    use rstest::rstest;

    const MATCHER_FACTORIES: &[fn(
        Vec<MatrixPattern>,
    ) -> Matcher<MatrixManyMatcher, MatrixNaiveManyMatcher>] = define_matcher_factories!(
        MatrixPattern,
        MatrixIndexingScheme,
        MatrixManyMatcher,
        MatrixNaiveManyMatcher
    );

    pub(super) fn apply_all_matchers(
        patterns: Vec<MatrixPattern>,
        subject: &MatrixString,
    ) -> [Vec<PatternMatch<MatrixPositionMap>>; 2] {
        // Skip the all deterministic matcher, too slow
        let all_matches = MATCHER_FACTORIES
            .iter()
            .map(|matcher_factory| {
                let matcher = matcher_factory(patterns.clone());
                if let Matcher::Many(matcher) = &matcher {
                    println!("{}", matcher.dot_string());
                }
                matcher.find_matches(subject).collect_vec()
            })
            .collect_vec();
        all_matches.into_iter().collect_vec().try_into().unwrap()
    }

    /// Do not compare min_pos and max_pos, as more than the pattern might have
    /// been matched
    pub(super) fn get_start_pos(
        matches: &[PatternMatch<MatrixPositionMap>],
    ) -> Vec<(PatternID, Option<MatrixSubjectPosition>)> {
        matches
            .iter()
            .map(|m| {
                let data = &m.match_data;
                (m.pattern, data.start_pos)
            })
            .collect()
    }

    #[rstest]
    #[case("f\n", vec!["f\n", "", "f\n"])]
    #[case("", vec!["--a$ca\n-\n$c\n", "\n---\n-\na\n", "\na---a\n", "-\n---a\n\n","---a-aaaa\n", "\n-a-\na\n", "\n--a\n-aa\n\n\n"])]
    #[case("", vec!["--a$ca\n-\n$c\n", "\n---\n-\na\n", "\na---a\n", "-\n---a\n\n"])]
    #[case("caaaabaa\n", vec!["-a--aaa\n", "-$c-$e--$d$c\na\naaaa\n", "a\n", "\n$c\n$ca\na-\n", "$d--ab\n$a$b$ca$aa\naaa\naaa\naaaaaaaa\na\n", "d-aaa\na\naa-a-a\n", "c----ba\n"])]
    #[case("aaa", vec!["a-a", "-aa", "aa-"])]
    #[case("aa\n", vec!["", ""])]
    #[case("aa\n", vec!["", "-"])]
    #[case("aab\n\n\n\n\n\n\naaaaaaaaa\n", vec![
        "----", "-------\n-$c\n\n\n\n\n\n---$c", "-\n-a"
    ])]
    fn proptest_fail_cases(#[case] subject: &str, #[case] patterns: Vec<&str>) {
        let subject = MatrixString::from(&subject);
        let patterns = patterns
            .into_iter()
            .map(MatrixPattern::parse_str)
            .collect_vec();

        let [default, naive] = apply_all_matchers(patterns, &subject);

        dbg!(&default);
        let mut default = get_start_pos(&default);
        let mut naive = get_start_pos(&naive);

        // dbg!(&all_matches);
        // println!(
        //     "{:?}",
        //     all_matches
        //         .iter()
        //         .map(|m| m.iter().map(|m| m.pattern).collect_vec())
        //         .collect_vec()
        // );

        default.sort();
        naive.sort();

        default.dedup();
        naive.dedup();

        assert_eq!(default, naive);
    }
}
