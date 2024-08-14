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
        matrix::{
            tests::{apply_all_matchers, clean_match_data},
            MatrixNaiveManyMatcher, MatrixString,
        },
        PortMatcher,
    };

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig {
        timeout: 200,
        .. ProptestConfig::default()
    })]
        #[test]
        fn proptest_matrix(
            subject in "([a-f]*\n)*",
            patterns in prop::collection::vec(any::<MatrixPattern>(), 1..10)
        ) {
            let subject = MatrixString::from(&subject);
            // Skip the all deterministic matcher, too slow
            let naive = MatrixNaiveManyMatcher::from_patterns(&patterns);
            let mut naive_matches = naive.find_matches(&subject).collect_vec();
            let [mut non_det, mut default] = apply_all_matchers(patterns, &subject);

            clean_match_data(&mut non_det);
            clean_match_data(&mut default);
            clean_match_data(&mut naive_matches);

            non_det.sort();
            default.sort();
            prop_assert_eq!(non_det, default);
        }
    }
}
