//!
//! The resulting data structure that is constructed for matching looks a lot
//! like a prefix tree. What makes this slightly more powerful (and interesting)
//! is that we can compare between characters at different positions in the
//! string using variables. This makes some transitions non-deterministic: for
//! example, the patterns "fooa" and "foo$x$x" match on the characters 0-2, but
//! the predicates for the 3rd position:
//!   1. character at the 3rd position is 'a' and
//!   2. character at the 3rd position equals character at the 4th position
//!      are not mutually exclusive. Indeed the string "fooaa" matches both patterns.
//!
//! Patterns are given by fixed-length strings, that are either a concrete string
//! such as "fooa" or may replace some characters with variables, such as
//! "foo$x$x". The name $x is a placeholder for any character, and the name can
//! be chosen arbitrarily.
//!
//! This is currently mostly useful for demonstration and testing purposes.
mod constraint;
mod pattern;
mod predicate;
#[cfg(feature = "proptest")]
mod proptest;

use std::{cmp, fmt::Debug};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    indexing::{BindVariableError, BindingResult, MissingIndexKeys, ValidBindings},
    IndexMap, IndexingScheme, ManyMatcher,
};
pub use constraint::StringConstraint;
pub use pattern::{CharVar, StringPattern};
pub use predicate::CharacterPredicate;

/// A matcher for strings.
pub type StringManyMatcher =
    ManyMatcher<StringPattern, StringPatternPosition, CharacterPredicate, StringIndexingScheme>;

/// Simple indexing scheme for strings.
///
/// As strings are sequences of characters, the indexing scheme is very simple:
/// just the integer position in the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StringIndexingScheme;

/// A map for string positions.
///
/// Only store the position of the start of the match, compute other positions by
/// adding offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum StringPositionMap {
    /// No key has been bound yet
    #[default]
    Unbound,
    /// Keys between 0 and `str_len` are valid and computed as offset from
    /// the start position
    Bound {
        /// The position of the start of the pattern in the string
        start_pos: StringSubjectPosition,
        /// The length of the substring matched so far
        str_len: usize,
    },
}

impl StringPositionMap {
    fn start_pos(&self) -> Option<StringSubjectPosition> {
        match self {
            Self::Unbound => None,
            &Self::Bound { start_pos, .. } => Some(start_pos),
        }
    }
}

impl IndexMap for StringPositionMap {
    type Key = StringPatternPosition;
    type Value = StringSubjectPosition;
    type ValueRef<'a> = StringSubjectPosition;

    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>> {
        let Self::Bound { start_pos, str_len } = self else {
            return None;
        };
        let StringPatternPosition(key) = var;
        if key < str_len {
            Some(StringSubjectPosition(start_pos.0 + key))
        } else {
            None
        }
    }

    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), BindVariableError> {
        let StringPatternPosition(key) = var;
        if key == 0 {
            if let Some(StringSubjectPosition(start_pos)) = self.start_pos() {
                return Err(BindVariableError::VariableExists {
                    key: key.to_string(),
                    curr_value: start_pos.to_string(),
                    new_value: format!("{:?}", val),
                });
            } else {
                *self = Self::Bound {
                    start_pos: val,
                    str_len: 1,
                };
            }
        } else {
            let Self::Bound { str_len, .. } = self else {
                return Err(BindVariableError::InvalidKey {
                    key: key.to_string(),
                });
            };
            *str_len = cmp::max(*str_len, key + 1);
        }
        Ok(())
    }
}

impl IndexingScheme<String> for StringIndexingScheme {
    type Map = StringPositionMap;

    fn valid_bindings(
        &self,
        key: &StringPatternPosition,
        known_bindings: &Self::Map,
        data: &String,
    ) -> BindingResult<Self, String> {
        let &StringPatternPosition(offset) = key;

        if offset == 0 {
            // No key has been bound yet
            assert!(matches!(known_bindings, Self::Map::Unbound));
            // For binding the start position, any string position is valid
            Ok(ValidBindings((0..data.len()).map_into().collect()))
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            let Some(StringSubjectPosition(start_pos)) = known_bindings.start_pos() else {
                return Err(MissingIndexKeys(vec![StringPatternPosition::start()]));
            };

            let str_pos = start_pos + offset;
            if str_pos < data.len() {
                Ok(ValidBindings(vec![(start_pos + offset).into()]))
            } else {
                Ok(ValidBindings(vec![]))
            }
        }
    }
}

/// A position in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StringSubjectPosition(usize);

/// An index key, obtained from the position index of the character in the pattern.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StringPatternPosition(usize);

impl StringPatternPosition {
    fn start() -> Self {
        Self(0)
    }
}

impl Debug for StringPatternPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "char@{}", self.0)
    }
}

#[cfg(test)]
pub(super) mod tests {

    use rstest::rstest;

    use self::pattern::StringPattern;
    use crate::{ManyMatcher, PatternMatch, PortMatcher};

    use super::*;

    fn non_det_matcher(patterns: Vec<StringPattern>) -> StringManyMatcher {
        ManyMatcher::from_patterns_with_det_heuristic(patterns, |_| false)
    }

    fn default_matcher(patterns: Vec<StringPattern>) -> StringManyMatcher {
        ManyMatcher::from_patterns(patterns)
    }

    fn det_matcher(patterns: Vec<StringPattern>) -> StringManyMatcher {
        ManyMatcher::from_patterns_with_det_heuristic(patterns, |_| true)
    }

    pub(super) const MATCHER_FACTORIES: &[fn(Vec<StringPattern>) -> StringManyMatcher] =
        &[non_det_matcher, default_matcher, det_matcher];

    #[test]
    fn test_string_matching() {
        let p1 = StringPattern::parse_str("ab$xcd$x");
        let p2 = StringPattern::parse_str("abcc");

        let matcher = StringManyMatcher::from_patterns(vec![p1, p2]);

        let result = matcher.find_matches(&"abccdc".to_string()).collect_vec();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dummy_len_3_string_matching() {
        let p1 = StringPattern::parse_str("$x$y$z");

        let matcher = StringManyMatcher::from_patterns(vec![p1]);

        let result = matcher.find_matches(&"ab".to_string()).collect_vec();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pattern_with_dummy_end() {
        let p1 = StringPattern::parse_str("$x$x$z");

        let matcher = StringManyMatcher::from_patterns(vec![p1]);

        let result = matcher.find_matches(&"aa".to_string()).collect_vec();

        assert_eq!(result.len(), 0);
    }

    pub(super) fn apply_all_matchers(
        patterns: Vec<StringPattern>,
        subject: &str,
    ) -> [Vec<PatternMatch<StringPositionMap>>; 3] {
        let all_matches = MATCHER_FACTORIES
            .iter()
            .map(|matcher_factory| {
                matcher_factory(patterns.clone())
                    .find_matches(&subject.to_string())
                    .collect_vec()
            })
            .collect_vec();
        all_matches.into_iter().collect_vec().try_into().unwrap()
    }

    #[rstest]
    #[case("h", vec!["h$aa", "$a$b"])]
    #[case("aa", vec!["a", "b"])]
    #[case("aa", vec!["a", "$c$c"])]
    #[case("haa", vec!["$aaa", "$a$b$c"])]
    #[case("xag", vec!["x$ag", "$aa"])]
    #[case("aaaayaazz", vec!["$a$baya", "$a$b$c$d$ez"])]
    #[case("aea", vec!["$d$aa", "$aea"])]
    #[case("eaaa", vec!["a", "$aa", "$b$a$ca", "$b$c$e$e"])]
    #[case("eaa", vec!["a$b$a", "$a$e$e", "b$c$a", "$c$aa"])]
    fn proptest_fail_cases(#[case] subject: &str, #[case] patterns: Vec<&str>) {
        use itertools::Itertools;

        let patterns = patterns
            .into_iter()
            .map(StringPattern::parse_str)
            .collect_vec();

        let [mut non_det, mut default, mut det] = apply_all_matchers(patterns, subject);

        // println!("{}", non_det_matcher(patterns.clone()).dot_string());
        // println!("{}", det_matcher(patterns.clone()).dot_string());
        // Compare results up to reordering
        non_det.sort();
        default.sort();
        det.sort();
        assert_eq!(non_det, default);
        assert_eq!(non_det, det);
    }
}
