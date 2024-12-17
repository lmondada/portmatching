use std::{borrow::Borrow, cmp, fmt::Debug};

use itertools::Itertools;

use crate::{
    indexing::IndexKey, pattern::Satisfiable, predicate::ConstraintLogic, ArityPredicate,
    Constraint, Predicate,
};

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

impl Predicate<String, StringSubjectPosition> for CharacterPredicate {
    fn check(&self, args: &[impl Borrow<StringSubjectPosition>], data: &String) -> bool {
        match self {
            CharacterPredicate::BindingEq => {
                let (StringSubjectPosition(pos1), StringSubjectPosition(pos2)) =
                    args.iter().map(|a| a.borrow()).collect_tuple().unwrap();
                let char1 = data.chars().nth(*pos1);
                let char2 = data.chars().nth(*pos2);
                char1.is_some() && char1 == char2
            }
            CharacterPredicate::ConstVal(c) => {
                let StringSubjectPosition(pos) =
                    args.iter().map(|a| a.borrow()).exactly_one().ok().unwrap();
                data.chars().nth(*pos) == Some(*c)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum BranchClass<K> {
    Position(K),
}

impl<K: IndexKey> ConstraintLogic<K> for CharacterPredicate {
    type BranchClass = BranchClass<K>;

    fn get_class(&self, keys: &[K]) -> Self::BranchClass {
        use CharacterPredicate::*;
        assert_eq!(self.arity(), keys.len());

        match self {
            BindingEq => {
                // Treat the predicate as an assignment of the larger key
                let max_key = cmp::max(keys[0], keys[1]);
                BranchClass::Position(max_key)
            }
            ConstVal(_) => BranchClass::Position(keys[0]),
        }
    }

    fn condition_on(
        &self,
        keys: &[K],
        condition: &Constraint<K, Self>,
    ) -> Satisfiable<Constraint<K, Self>> {
        assert_eq!(self.get_class(keys), condition.get_class());

        // Safe as all predicates have arity >= 1
        let k0 = keys[0];
        let cond_k0 = condition.required_bindings()[0];

        use CharacterPredicate::*;
        match (self, condition.predicate()) {
            (ConstVal(a), ConstVal(b)) if a == b => Satisfiable::Tautology,
            (ConstVal(a), ConstVal(b)) => Satisfiable::No,
            (BindingEq, BindingEq) if k0 == cond_k0 => Satisfiable::Tautology,
            _ => {
                // TODO: if we had access to other conditions, we could also
                // solve these cases,
                // i.e. BindingEq(c1, c2) + ConstVal(c2) => ConstVal(c1)
                // for now: pretend we don't know anything
                Satisfiable::Yes(self.try_into_constraint(keys.to_vec()).unwrap())
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

#[cfg(test)]
mod tests {
    use crate::{concrete::string::StringPattern, Pattern};

    #[test]
    fn test_to_constraints() {
        let p = StringPattern::parse_str("$c$d$eca$c$c$aaaba");
        p.into_logic(); // TODO: turn into a Result.unwrap()
    }
}
