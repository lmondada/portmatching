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

    use crate::{
        string::tests::{pattern_match_eq, MATCHER_FACTORIES},
        PortMatcher,
    };

    use super::*;

    proptest! {
        #[test]
        fn proptest_string(
            subject in "[a-f]*",
            patterns in prop::collection::vec(any::<StringPattern>(), 1..10)
        ) {
            let all_matches = MATCHER_FACTORIES.iter().map(|matcher_factory| {
                matcher_factory(patterns.clone()).find_matches(&subject).collect_vec()
            });
            let Some((exp, act1, act2)) = all_matches.into_iter().collect_tuple() else {
                panic!("Expected 3 matchers");
            };
            prop_assert!(pattern_match_eq(&exp, &act1));
            prop_assert!(pattern_match_eq(&exp, &act2));
        }
    }
}