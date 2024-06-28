use std::fmt::Debug;

use itertools::Itertools;

use crate::{ArityPredicate, Predicate};

use super::StringPosition;

/// A predicate for matching a string.
///
/// We support two types of predicates: matching a character against a constant
/// `char`, and comparing to character variables.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StringPredicate {
    /// Comparison between two bound values
    BindingEq,
    /// Comparison of a bound value with a constant char
    ConstVal(char),
}

impl ArityPredicate for StringPredicate {
    fn arity(&self) -> usize {
        match self {
            StringPredicate::BindingEq => 2,
            StringPredicate::ConstVal(_) => 1,
        }
    }
}

impl Predicate<String> for StringPredicate {
    type Value = StringPosition;

    fn check(&self, data: &String, args: &[&StringPosition]) -> bool {
        match self {
            StringPredicate::BindingEq => {
                let (StringPosition(pos1), StringPosition(pos2)) =
                    args.iter().collect_tuple().unwrap();
                let char1 = data.chars().nth(*pos1);
                let char2 = data.chars().nth(*pos2);
                char1.is_some() && char1 == char2
            }
            StringPredicate::ConstVal(c) => {
                let StringPosition(pos) = args.iter().exactly_one().unwrap();
                data.chars().nth(*pos) == Some(*c)
            }
        }
    }
}

impl Debug for StringPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BindingEq => write!(f, "VariableEq"),
            Self::ConstVal(c) => write!(f, "Const[{}]", c),
        }
    }
}
