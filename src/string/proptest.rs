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

    use crate::string::tests::apply_all_matchers;

    use super::*;

    proptest! {
        #[test]
        fn proptest_string(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..10)
        ) {
            let [mut nondet, mut default, mut det] = apply_all_matchers(patterns, &subject);

            nondet.sort();
            default.sort();
            det.sort();
            prop_assert_eq!(&nondet, &default);
            prop_assert_eq!(&nondet, &det);
        }
    }
}
