//!
//! The resulting data structure that is constructed for matching looks a lot
//! like a prefix tree. What makes this slightly more powerful (and interesting)
//! is that we can compare between characters at different positions in the
//! string using variables. This makes some transitions non-deterministic: for
//! example, the patterns "fooa" and "foo$x$x" match on the characters 0-2, but
//! the predicates for the 3rd position:
//!   1. character at the 3rd position is 'a' and
//!   2. character at the 3rd position equals character at the 4th position
//! are not mutually exclusive. Indeed the string "fooaa" matches both patterns.
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

use std::fmt::Debug;

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
    ManyMatcher<StringPattern, StringIndexKey, CharacterPredicate, StringIndexingScheme>;

/// Simple indexing schemes for strings.
///
/// As strings form a chain of characters, the indexing scheme is very simple:
/// just the integer position in the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StringIndexingScheme;

/// A map for string positions.
///
/// Only store the position of the root, compute other positions by adding
/// offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct StringPositionMap(Option<usize>);
impl IndexMap for StringPositionMap {
    type Key = StringIndexKey;
    type Value = StringPosition;
    type ValueRef<'a> = StringPosition;

    fn get(&self, var: &Self::Key) -> Option<Self::ValueRef<'_>> {
        let StringPositionMap(Some(root)) = self else {
            return None;
        };
        let StringIndexKey(key) = var;
        Some(StringPosition(root + key))
    }

    fn bind(&mut self, var: Self::Key, val: Self::Value) -> Result<(), BindVariableError> {
        let StringIndexKey(key) = var;
        if key == 0 {
            if let Some(curr_value) = self.0 {
                return Err(BindVariableError::VariableExists {
                    key: key.to_string(),
                    curr_value: curr_value.to_string(),
                    new_value: format!("{:?}", val),
                });
            } else {
                self.0 = Some(val.0);
            }
        } else {
        }
        Ok(())
    }
}

impl IndexingScheme<String> for StringIndexingScheme {
    type Map = StringPositionMap;

    fn valid_bindings(
        &self,
        key: &StringIndexKey,
        known_bindings: &Self::Map,
        data: &String,
    ) -> BindingResult<Self, String> {
        if key == &StringIndexKey::root() {
            // For the root binding, any string position is valid
            Ok(ValidBindings((0..data.len()).map_into().collect()))
        } else {
            let Some(StringPosition(root_pos)) = known_bindings.get(&StringIndexKey::root()) else {
                return Err(MissingIndexKeys(vec![StringIndexKey::root()]));
            };

            let &StringIndexKey(offset) = key;

            Ok(ValidBindings(vec![(root_pos + offset).into()]))
        }
    }
}

/// A position in a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StringPosition(usize);

/// An index key, obtained from the position index of the character in the pattern.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, From, Into)]
pub struct StringIndexKey(usize);

impl StringIndexKey {
    fn root() -> Self {
        Self(0)
    }
}

impl Debug for StringIndexKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "char@{}", self.0)
    }
}

#[cfg(test)]
pub(super) mod tests {

    use rstest::rstest;

    use self::pattern::StringPattern;
    use crate::{ManyMatcher, PortMatcher};

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
        let all_matches = MATCHER_FACTORIES
            .iter()
            .map(|matcher_factory| {
                matcher_factory(patterns.clone())
                    .find_matches(&subject.to_string())
                    .collect_vec()
            })
            .collect_vec();

        // dbg!(&all_matches);
        // println!("{}", non_det_matcher(patterns.clone()).dot_string());
        // println!("{}", det_matcher(patterns.clone()).dot_string());
        let Some((mut exp, mut act1, mut act2)) = all_matches.into_iter().collect_tuple() else {
            panic!("Expected 3 matchers");
        };
        // Compare results up to reordering
        exp.sort();
        act1.sort();
        act2.sort();
        assert_eq!(exp, act1);
        assert_eq!(exp, act2);
    }
}
