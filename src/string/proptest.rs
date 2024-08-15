use proptest::prelude::*;

use super::pattern::StringPattern;

const REGEXP: &str = r"(\$?[a-f])*";

prop_compose! {
    fn arb_string_pattern()(str_pat in REGEXP) -> StringPattern {
        StringPattern::parse_str(&str_pat)
    }
}

impl Arbitrary for StringPattern {
    type Parameters = ();
    type Strategy = BoxedStrategy<StringPattern>;

    fn arbitrary_with(_: ()) -> Self::Strategy {
        arb_string_pattern().boxed()
    }
}

#[cfg(test)]
mod tests {

    use itertools::Itertools;

    use crate::string::tests::{apply_all_matchers, clean_match_data};

    use super::*;

    proptest! {
        #[test]
        #[ignore = "a bit slow"]
        fn proptest_string(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..5)
        ) {
            let (mut nondet, mut default, mut det) = apply_all_matchers(patterns, &subject, 0..3)
                .collect_tuple()
                .unwrap();

            clean_match_data(&mut nondet);
            clean_match_data(&mut default);
            clean_match_data(&mut det);

            nondet.sort();
            default.sort();
            det.sort();

            prop_assert_eq!(&nondet, &default);
            prop_assert_eq!(&nondet, &det);
        }

        #[test]
        #[ignore = "a bit slow"]
        fn proptest_string_large(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..20)
        ) {
            let (mut nondet, mut default) = apply_all_matchers(patterns, &subject, 0..2)
                .collect_tuple()
                .unwrap();

            clean_match_data(&mut nondet);
            clean_match_data(&mut default);

            nondet.sort();
            default.sort();

            prop_assert_eq!(&nondet, &default);
        }
    }
}
