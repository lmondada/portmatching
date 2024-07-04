use proptest::prelude::*;

use super::pattern::MatrixPattern;

const REGEXP: &str = r"((-|(\$?[a-f])){0,10}\n){0,10}";

prop_compose! {
    fn arb_matrix_pattern()(str_pat in REGEXP) -> MatrixPattern {
        MatrixPattern::parse_str(&str_pat)
    }
}

impl Arbitrary for MatrixPattern {
    type Parameters = ();
    type Strategy = BoxedStrategy<MatrixPattern>;

    fn arbitrary_with(_: ()) -> Self::Strategy {
        arb_matrix_pattern().boxed()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::{
        matrix::{MatrixIndexKey, MatrixManyMatcher, MatrixPosition, MatrixString},
        string::tests::define_matcher_factories,
        HashMap, PatternMatch, PortMatcher,
    };

    use super::*;

    const MATCHER_FACTORIES: &[fn(Vec<MatrixPattern>) -> MatrixManyMatcher] =
        define_matcher_factories!(MatrixPattern, MatrixManyMatcher);

    /// Comparing two lists of pattern matches.
    ///
    /// We cannot use equality for two reasons:
    ///  - we ignore the ordering of the matches
    ///  - not all keys may be bound depending on the order things were matched
    ///
    /// The simplest way to identify a match is with the triple (PatternID, start_row, start_col).
    pub(super) fn pattern_match_eq(
        expected: &[PatternMatch<HashMap<MatrixIndexKey, MatrixPosition>>],
        actual: &[PatternMatch<HashMap<MatrixIndexKey, MatrixPosition>>],
    ) -> bool {
        let get_match_key =
            |PatternMatch {
                 pattern,
                 match_data,
             }: &PatternMatch<HashMap<MatrixIndexKey, MatrixPosition>>| {
                let Some((MatrixIndexKey(row_offset, col_offset), MatrixPosition(row, col))) =
                    match_data.iter().next()
                else {
                    panic!("Empty match data");
                };
                (*pattern, row - row_offset, col - col_offset)
            };
        // Build a map from pattern ID to all positions where it matches
        let get_slice_key = |matches: &[PatternMatch<_>]| {
            let mut keys = matches.iter().map(get_match_key).fold(
                HashMap::<_, Vec<_>>::default(),
                |mut acc, (pattern, row, col)| {
                    acc.entry(pattern).or_default().push((row, col));
                    acc
                },
            );
            keys.values_mut().for_each(|vec| vec.sort());
            keys
        };
        get_slice_key(expected) == get_slice_key(actual)
    }

    proptest! {
        #[test]
        fn proptest_matrix(
            subject in "([a-f]*\n)*",
            patterns in prop::collection::vec(any::<MatrixPattern>(), 1..10)
        ) {
            // println!("{}", MATCHER_FACTORIES[0](patterns.clone()).dot_string());
            let dot_string = MATCHER_FACTORIES[1](patterns.clone()).dot_string();
            std::fs::write("matrix_matcher_graph.dot", dot_string).expect("Failed to write dot file");

            let subject = MatrixString::from(&subject);
            // Skip the all deterministic matcher, too slow
            let all_matches = MATCHER_FACTORIES[..2].iter().map(|matcher_factory| {
                matcher_factory(patterns.clone()).find_matches(&subject).collect_vec()
            }).collect_vec();
            // dbg!(&all_matches);
            let Some((exp, act)) = all_matches.into_iter().collect_tuple() else {
                panic!("Expected 2 matchers");
            };
            prop_assert!(pattern_match_eq(&exp, &act));
        }
    }
}
