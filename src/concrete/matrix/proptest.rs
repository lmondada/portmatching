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

    use crate::concrete::matrix::{
        tests::{apply_all_matchers, get_start_pos},
        MatrixString,
    };

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig {
        timeout: 200,
        .. ProptestConfig::default()
    })]
        #[test]
        #[ignore = "a bit slow"]
        fn proptest_matrix(
            subject in "([a-f]*\n)*",
            patterns in prop::collection::vec(any::<MatrixPattern>(), 1..10)
        ) {
            let subject = MatrixString::from(&subject);
            // Skip the all deterministic matcher, too slow
            let [default, naive] = apply_all_matchers(patterns, &subject);

            let mut default = get_start_pos(&default);
            let mut naive = get_start_pos(&naive);

            default.sort();
            naive.sort();

            default.dedup();
            naive.dedup();

            prop_assert_eq!(default, naive);
        }
    }
}
