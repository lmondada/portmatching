use std::{borrow::Borrow, fmt::Debug};

use itertools::Itertools;

use crate::{ArityPredicate, Predicate};

use super::StringSubjectPosition;

/// Predicate used when matching strings.
///
/// We support two types of predicates: matching a character against a constant
/// `char`, and comparing to character variables.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CharacterPredicate {
    /// Comparison between two bound values
    BindingEq,
    /// Comparison of a bound value with a constant char
    ConstVal(char),
}

impl ArityPredicate for CharacterPredicate {
    fn arity(&self) -> usize {
        match self {
            CharacterPredicate::BindingEq => 2,
            CharacterPredicate::ConstVal(_) => 1,
        }
    }
}

impl Predicate<String> for CharacterPredicate {
    type InvalidPredicateError = String;

    fn check(
        &self,
        data: &String,
        args: &[impl Borrow<StringSubjectPosition>],
    ) -> Result<bool, String> {
        match self {
            CharacterPredicate::BindingEq => {
                let (StringSubjectPosition(pos1), StringSubjectPosition(pos2)) =
                    args.iter().map(|a| a.borrow()).collect_tuple().unwrap();
                let char1 = data.chars().nth(*pos1);
                let char2 = data.chars().nth(*pos2);
                Ok(char1.is_some() && char1 == char2)
            }
            CharacterPredicate::ConstVal(c) => {
                let StringSubjectPosition(pos) =
                    args.iter().map(|a| a.borrow()).exactly_one().ok().unwrap();
                Ok(data.chars().nth(*pos) == Some(*c))
            }
        }
    }
}

impl Debug for CharacterPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BindingEq => write!(f, "VariableEq"),
            Self::ConstVal(c) => write!(f, "Const[{}]", c),
        }
    }
}
