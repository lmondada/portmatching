//! Pattern matching on 2D character grids.
//!
//! This is a straightforward generalisation of the string matcher to 2D. It
//! is interesting for testing and demonstration as overlapping but not contained
//! matches are possible.
use std::{
    borrow::Borrow,
    collections::BTreeMap,
    fmt::{self, Debug},
};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    concrete::string::CharacterPredicate,
    constraint::{DeterministicConstraintSelector, EvaluatePredicate},
    indexing::{Binding, IndexedData},
    BindMap, IndexingScheme, ManyMatcher, NaiveManyMatcher,
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

type BranchSelector = DeterministicConstraintSelector<MatrixPatternPosition, CharacterPredicate>;

/// A matcher for 2D character matrices.
pub type MatrixManyMatcher = ManyMatcher<MatrixPattern, MatrixPatternPosition, BranchSelector>;
/// A naive matcher for 2D character matrices.
///
/// Only use for testing, as it is too slow for practical purposes.
pub type MatrixNaiveManyMatcher = NaiveManyMatcher<MatrixPatternPosition, BranchSelector>;

impl EvaluatePredicate<MatrixString, MatrixSubjectPosition> for CharacterPredicate {
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

type MatrixBindMap = BTreeMap<MatrixPatternPosition, Option<MatrixSubjectPosition>>;

/// Simple indexing schemes for matrices.
///
/// The indexing scheme is a pair of (row, col) positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixIndexingScheme;

impl IndexingScheme for MatrixIndexingScheme {
    type BindMap = MatrixBindMap;
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
        known_bindings: &MatrixBindMap,
    ) -> Vec<MatrixSubjectPosition> {
        let &MatrixPatternPosition(key_row, key_col) = key;

        if key_row == 0 && key_col == 0 {
            // No key has been bound yet
            assert!(known_bindings.is_empty());

            // For the root binding, any matrix position is valid
            (0..self.rows.len())
                .flat_map(|row| (0..self.rows[row].len()).map(move |col| (row, col)))
                .map_into()
                .collect()
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            let (start_row, start_col) = match known_bindings
                .get_binding(&MatrixPatternPosition::default())
                .borrowed()
            {
                Binding::Bound(&MatrixSubjectPosition(start_row, start_col)) => {
                    (start_row, start_col)
                }
                Binding::Failed => return vec![],
                Binding::Unbound => {
                    panic!("root binding not found. Always bind the start position first");
                }
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

    type MatcherFactory =
        fn(Vec<MatrixPattern>) -> Matcher<MatrixManyMatcher, MatrixNaiveManyMatcher>;
    const MATCHER_FACTORIES: &[MatcherFactory] = define_matcher_factories!(
        MatrixPattern,
        MatrixIndexingScheme,
        MatrixManyMatcher,
        MatrixNaiveManyMatcher
    );

    pub(super) fn apply_all_matchers(
        patterns: Vec<MatrixPattern>,
        subject: &MatrixString,
    ) -> [Vec<PatternMatch<MatrixBindMap>>; 2] {
        // Skip the all deterministic matcher, too slow
        let all_matches = MATCHER_FACTORIES
            .iter()
            .map(|matcher_factory| {
                let matcher = matcher_factory(patterns.clone());
                // if let Matcher::Many(matcher) = &matcher {
                //     println!("{}", matcher.dot_string());
                // }
                matcher.find_matches(subject).collect_vec()
            })
            .collect_vec();
        all_matches.into_iter().collect_vec().try_into().unwrap()
    }

    /// Do not compare min_pos and max_pos, as more than the pattern might have
    /// been matched
    pub(super) fn get_start_pos(
        matches: &[PatternMatch<MatrixBindMap>],
    ) -> Vec<(PatternID, Binding<MatrixSubjectPosition>)> {
        matches
            .iter()
            .map(|m| {
                let data = &m.match_data;
                (
                    m.pattern,
                    data.get_binding(&MatrixPatternPosition::start()).copied(),
                )
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
    #[case("aaaaac\naaaaaafa\n\n\n\naaaaaaaaaaa\n", vec![
        "\n$a--a$a\n\n\n\n$a", "--ca-\nb--fa"
    ])]
    fn proptest_fail_cases(#[case] subject: &str, #[case] patterns: Vec<&str>) {
        let subject = MatrixString::from(&subject);
        let patterns = patterns
            .into_iter()
            .map(MatrixPattern::parse_str)
            .collect_vec();

        let [default, naive] = apply_all_matchers(patterns, &subject);

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
