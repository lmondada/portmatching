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

use std::{borrow::Borrow, cmp, fmt::Debug};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    branch_selector::DisplayBranchSelector,
    indexing::{BindVariableError, Binding, IndexedData},
    predicate::PredicatePatternDefaultSelector,
    BindMap, IndexingScheme, ManyMatcher,
};
pub use constraint::StringConstraint;
pub use pattern::{CharVar, StringPattern};
pub use predicate::{BranchClass, CharacterPredicate};

type BranchSelector = PredicatePatternDefaultSelector<StringPatternPosition, CharacterPredicate>;

impl DisplayBranchSelector for BranchSelector {
    fn fmt_class(&self) -> String {
        let Some(cls) = self.get_class() else {
            return String::new();
        };
        match cls {
            BranchClass::Position(pos) => format!("Position({pos:?})"),
        }
    }

    fn fmt_nth_constraint(&self, n: usize) -> String {
        let pred = &self.predicates()[n];
        match pred {
            CharacterPredicate::BindingEq => format!(" == {:?}", self.keys(n)[0]),
            CharacterPredicate::ConstVal(c) => format!(" == {:?}", c),
        }
    }
}

/// A matcher for strings.
pub type StringManyMatcher = ManyMatcher<StringPattern, StringPatternPosition, BranchSelector>;

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
        /// The offset from which bindings have failed
        failed_pos: Option<usize>,
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

impl BindMap for StringPositionMap {
    type Key = StringPatternPosition;
    type Value = StringSubjectPosition;

    fn get_binding(&self, var: &Self::Key) -> Binding<impl Borrow<Self::Value> + '_> {
        let &Self::Bound {
            start_pos,
            str_len,
            failed_pos,
        } = self
        else {
            return Binding::Unbound;
        };
        let &StringPatternPosition(key) = var;
        if key < str_len {
            Binding::Bound(StringSubjectPosition(start_pos.0 + key))
        } else if failed_pos.is_some() && key >= failed_pos.unwrap() {
            Binding::Failed
        } else {
            Binding::Unbound
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
                    failed_pos: None,
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

    fn bind_failed(&mut self, var: Self::Key) {
        let StringPatternPosition(key) = var;
        let Self::Bound { failed_pos, .. } = self else {
            panic!("Always bind the start position first, which cannot fail");
        };
        if let Some(failed_pos) = failed_pos.as_mut() {
            *failed_pos = cmp::min(*failed_pos, key);
        } else {
            *failed_pos = Some(key);
        }
    }
}

impl IndexingScheme for StringIndexingScheme {
    type BindMap = StringPositionMap;
    type Key = StringPatternPosition;
    type Value = StringSubjectPosition;

    fn required_bindings(&self, key: &Self::Key) -> Vec<Self::Key> {
        let &StringPatternPosition(offset) = key;

        if offset == 0 {
            vec![]
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            vec![StringPatternPosition::start()]
        }
    }
}

impl IndexedData<StringPatternPosition> for String {
    type IndexingScheme = StringIndexingScheme;
    type Value = <Self::IndexingScheme as IndexingScheme>::Value;
    type BindMap = <Self::IndexingScheme as IndexingScheme>::BindMap;

    fn list_bind_options(
        &self,
        key: &StringPatternPosition,
        known_bindings: &Self::BindMap,
    ) -> Vec<Self::Value> {
        let &StringPatternPosition(offset) = key;

        if offset == 0 {
            // For binding the start position, any string position is valid
            (0..self.len()).map_into().collect()
        } else {
            // Must bind the start position first; all other positions are
            // obtained by offsetting from start
            let Some(StringSubjectPosition(start_pos)) = known_bindings.start_pos() else {
                return vec![];
            };

            let str_pos = start_pos + offset;
            if str_pos < self.len() {
                vec![(start_pos + offset).into()]
            } else {
                vec![]
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
    /// The start position of the string.
    pub fn start() -> Self {
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

    use std::ops::Range;

    use auto_enums::auto_enum;
    use rstest::rstest;

    use self::pattern::StringPattern;
    use crate::{NaiveManyMatcher, PatternID, PatternMatch, PortMatcher};

    use super::*;

    #[derive(Debug, Clone, From)]
    enum Matcher {
        Many(StringManyMatcher),
        Naive(NaiveManyMatcher<StringPatternPosition, BranchSelector>),
    }

    impl PortMatcher<String> for Matcher {
        type Match = StringPositionMap;

        #[auto_enum(Iterator)]
        fn find_matches<'a>(
            &'a self,
            host: &'a String,
        ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
            match self {
                Matcher::Many(m) => m.find_matches(host),
                Matcher::Naive(m) => m.find_matches(host),
            }
        }
    }

    macro_rules! define_matcher_factories {
        ($StringPattern:ty, $StringManyMatcher:ty) => {{
            fn many_matcher(patterns: Vec<$StringPattern>) -> Matcher {
                <$StringManyMatcher>::from_patterns::<StringIndexingScheme>(patterns).into()
            }

            fn naive_matcher(patterns: Vec<$StringPattern>) -> Matcher {
                Matcher::Naive(NaiveManyMatcher::from_patterns::<StringIndexingScheme, _>(
                    patterns,
                ))
            }

            &[many_matcher, naive_matcher]
        }};
    }

    pub(crate) use define_matcher_factories;

    #[test]
    fn test_string_matching() {
        let p1 = StringPattern::parse_str("ab$xcd$x");
        let p2 = StringPattern::parse_str("abcc");

        let matcher = StringManyMatcher::from_patterns::<StringIndexingScheme>(vec![p1, p2]);

        let result = matcher.find_matches(&"abccdc".to_string()).collect_vec();

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_dummy_len_3_string_matching() {
        let p1 = StringPattern::parse_str("$x$y$z");

        let matcher = StringManyMatcher::from_patterns::<StringIndexingScheme>(vec![p1]);

        let result = matcher.find_matches(&"ab".to_string()).collect_vec();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_empty_pattern() {
        let p1 = StringPattern::parse_str("");

        let matcher = StringManyMatcher::from_patterns::<StringIndexingScheme>(vec![p1]);

        let result = matcher.find_matches(&"ab".to_string()).collect_vec();

        assert_eq!(result, [(PatternID(0), StringPositionMap::Unbound).into()]);
    }

    #[test]
    fn test_pattern_with_dummy_end() {
        let p1 = StringPattern::parse_str("$x$x$z");

        let matcher = StringManyMatcher::from_patterns::<StringIndexingScheme>(vec![p1]);

        let result = matcher.find_matches(&"aa".to_string()).collect_vec();

        assert_eq!(result.len(), 0);
    }

    const MATCHER_FACTORIES: &[fn(Vec<StringPattern>) -> Matcher] =
        define_matcher_factories!(StringPattern, StringManyMatcher);

    pub(super) fn apply_all_matchers(
        patterns: Vec<StringPattern>,
        subject: &str,
    ) -> impl Iterator<Item = Vec<PatternMatch<StringPositionMap>>> + '_ {
        MATCHER_FACTORIES.iter().map(move |matcher_factory| {
            let matcher = matcher_factory(patterns.clone());
            dbg!(&matcher);
            if let Matcher::Many(m) = &matcher {
                println!("{}", m.dot_string());
            }
            matcher.find_matches(&subject.to_string()).collect_vec()
        })
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
    #[case("aaac", vec!["$d$d", "bac"])]
    #[case("", vec!["aaaaaa", "baa", "a$b$a$a$c", "$c$b$b"])]
    #[case("", vec!["caa", "a$e$a$b$fa$cca$aa", "$c$d$eca$c$c$aaaba", "$baaaabaaaaaaaaa$caaa", "ea$a$b$caa", "aaaaa", ""])]
    #[case("fef", vec!["$b$a", "fe", "$ae"])]
    #[case("", vec!["$aea", "a$a", ""])]
    #[case("", vec!["aaaaaaaaaaaaaaaaaaaa$a", "$aea", "a$baab", "$b$bbaaaaa", "", "b"])]
    #[case("", vec![
        "$a$df$d$d$c$c$e$b",
        "$ac$c$c$f$f$cf$d$aa",
        "$c$b$bbdbaae$caef$afe$bb$f$c$f$d$f$b",
        "e$e$c$cf$e$ca$c$f$cc$bc$ef",
        "$fc$f$ad$a$e$b$c$e$feab",
    ])]
    #[case("baaa", vec![
        "b$c$caaaaaaaaaaaa$b$daa$aa",
        "d$aadaaaaa$baaaa$faaaaadaa$eaaa$caa",
        "dcbaaaaaaaaaaaaaaaaa$aaa$d$ba",
        "$c$daaaaaaaaaaaa$fa$ea",
        "$a$b$c",
        "cb$aaaaaaaaaaaaaaaaa$ca$b$eaaaa$daaa",
        "$a$bbaaaaaaa",
        "aa$faaaaa",
    ])]
    #[case("", vec![
        "deabaaaaaaaa$aa",
        "acaaaaaaa$da$aaa$baaa$caaaa",
        "$ce$aaa",
        "$b$baa",
        "aaaaaaa",
        "aa$aaaaaaaa$baa$caaa",
        "aeaaaaaaaaaaaaaaaaaaa$aa",
        "baaaaaaaaaaaa$aaa$ba",
        "$e$aa$eaaaa",
        "ceaaaaaaaaaa$baa$a$caa$daa",
        "b$aaaa",
    ])]
    #[case("abe", vec![
        "$a",
        "a$ae",
        "b",
        "aba",
        "$cb$a",
    ])]
    fn proptest_fail_cases(#[case] subject: &str, #[case] patterns: Vec<&str>) {
        let patterns = patterns
            .into_iter()
            .map(StringPattern::parse_str)
            .collect_vec();

        // let [mut non_det, mut default, mut det] = apply_all_matchers(patterns, subject);
        let (mut default, mut naive) = apply_all_matchers(patterns, subject)
            .collect_tuple()
            .unwrap();

        // Compare results up to reordering
        default.sort();
        naive.sort();

        assert_eq!(default, naive);
    }
}
