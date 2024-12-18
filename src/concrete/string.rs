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
mod pattern;
mod predicate;
#[cfg(feature = "proptest")]
mod proptest;

use std::{
    borrow::Borrow,
    cmp,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
};

use derive_more::{From, Into};
use itertools::Itertools;

use crate::{
    branch_selector::DisplayBranchSelector,
    indexing::{BindVariableError, Binding, IndexKey, IndexedData},
    predicate::DeterministicPredicatePatternSelector,
    BindMap, IndexingScheme, ManyMatcher, NaiveManyMatcher,
};
pub use pattern::{CharVar, StringPattern};
pub use predicate::{BranchClass, CharacterPredicate};

type BranchSelector =
    DeterministicPredicatePatternSelector<StringPatternPosition, CharacterPredicate>;

/// A constraint for matching a string using [StringPredicate]s.
type StringConstraint = crate::Constraint<StringPatternPosition, CharacterPredicate>;

impl<K: IndexKey> DisplayBranchSelector
    for DeterministicPredicatePatternSelector<K, CharacterPredicate>
{
    fn fmt_class(&self) -> String {
        let Some(cls) = self.get_class() else {
            return String::new();
        };
        match cls {
            BranchClass::Position(pos) => format!("Position({pos:?})"),
        }
    }

    fn fmt_nth_constraint(&self, n: usize) -> String {
        match &self.predicates()[n] {
            CharacterPredicate::BindingEq => format!(" == {:?}", self.keys(n)[0]),
            CharacterPredicate::ConstVal(c) => format!(" == {:?}", c),
        }
    }
}

/// A matcher for strings.
pub type StringManyMatcher = ManyMatcher<StringPattern, StringPatternPosition, BranchSelector>;
pub type StringNaiveManyMatcher = NaiveManyMatcher<StringPatternPosition, BranchSelector>;

/// Simple indexing scheme for strings.
///
/// As strings are sequences of characters, the indexing scheme is very simple:
/// just the integer position in the string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StringIndexingScheme;

type StringBindMap = BTreeMap<StringPatternPosition, Option<StringSubjectPosition>>;

impl IndexingScheme for StringIndexingScheme {
    type BindMap = StringBindMap;
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
            let Binding::Bound(StringSubjectPosition(start_pos)) = known_bindings
                .get_binding(&StringPatternPosition::start())
                .copied()
            else {
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
    use crate::{Pattern, PatternID, PatternMatch, PortMatcher};

    use super::*;

    #[derive(Debug, Clone)]
    pub(crate) enum Matcher<Many, Naive> {
        Many(Many),
        Naive(Naive),
    }

    impl<D, Many, Naive> PortMatcher<D> for Matcher<Many, Naive>
    where
        Many: PortMatcher<D>,
        Naive: PortMatcher<D, Match = Many::Match>,
    {
        type Match = Many::Match;

        #[auto_enum(Iterator)]
        fn find_matches<'a>(
            &'a self,
            host: &'a D,
        ) -> impl Iterator<Item = PatternMatch<Self::Match>> + 'a {
            match self {
                Matcher::Many(m) => m.find_matches(host),
                Matcher::Naive(m) => m.find_matches(host),
            }
        }
    }

    macro_rules! define_matcher_factories {
        ($Pattern:ty, $Scheme:ty, $ManyMatcher:ty, $NaiveMatcher:ty) => {{
            use crate::concrete::string::tests::Matcher;

            fn many_matcher(patterns: Vec<$Pattern>) -> Matcher<$ManyMatcher, $NaiveMatcher> {
                Matcher::Many(<$ManyMatcher>::from_patterns::<$Scheme>(patterns))
            }

            fn naive_matcher(patterns: Vec<$Pattern>) -> Matcher<$ManyMatcher, $NaiveMatcher> {
                Matcher::Naive(<$NaiveMatcher>::from_patterns::<$Scheme, _>(patterns))
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

        assert_eq!(result, [(PatternID(0), BTreeMap::default()).into()]);
    }

    #[test]
    fn test_pattern_with_dummy_end() {
        let p1 = StringPattern::parse_str("$x$x$z");

        let matcher = StringManyMatcher::from_patterns::<StringIndexingScheme>(vec![p1]);

        let result = matcher.find_matches(&"aa".to_string()).collect_vec();

        assert_eq!(result.len(), 0);
    }

    const MATCHER_FACTORIES: &[fn(
        Vec<StringPattern>,
    ) -> Matcher<StringManyMatcher, StringNaiveManyMatcher>] = define_matcher_factories!(
        StringPattern,
        StringIndexingScheme,
        StringManyMatcher,
        StringNaiveManyMatcher
    );

    pub(super) fn apply_all_matchers(
        patterns: Vec<StringPattern>,
        subject: &str,
    ) -> impl Iterator<Item = Vec<PatternMatch<StringBindMap>>> + '_ {
        MATCHER_FACTORIES.iter().map(move |matcher_factory| {
            let matcher = matcher_factory(patterns.clone());
            // if let Matcher::Many(m) = &matcher {
            //     println!("{}", m.dot_string());
            // }
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
    #[case("aaa", vec!["a$ba", "$aa$a", "aa$a"])]
    #[case("cbaaaaaab", vec![
        "c", "d", "$c$baaaaaa$baaaaaaaaaa","$baaaa$b$baaaa"
    ])]
    #[case("ddaaaaaadaaaaaaaaaaaaaad", vec![
        "$aaaaaaaabaaaaaaaaaaaaa$a$a", "$f$f", "$caaaa$caa$c",
    ])]
    #[case("aaaaaaaa", vec![
        "aaaaabaaa",
        "$c$bbaa$b$b",
    ])]
    #[case("aaaaabdd", vec![
        "aba$aa",
        "abaaa",
        "$abaaa",
        "$e$e$c$b$b",
        "$aaaba$a",
        "abaaa",
    ])]
    #[case("c", vec!["c", "$b$b"])]
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
