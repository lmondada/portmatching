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

    use crate::concrete::string::tests::apply_all_matchers;

    use super::*;

    proptest! {
        #[test]
        #[ignore = "a bit slow"]
        fn proptest_string(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..5)
        ) {
            let (mut default, mut naive) = apply_all_matchers(patterns, &subject)
                .collect_tuple()
                .unwrap();

            default.sort();
            naive.sort();

            prop_assert_eq!(&default, &naive);
        }

        #[test]
        #[ignore = "a bit slow"]
        fn proptest_string_large(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..20)
        ) {
            let (mut default, mut naive) = apply_all_matchers(patterns, &subject)
                .collect_tuple()
                .unwrap();

            default.sort();
            naive.sort();

            prop_assert_eq!(&default, &naive);
        }
    }
}
