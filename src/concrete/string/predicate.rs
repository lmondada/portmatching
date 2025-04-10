use std::{
    borrow::Borrow,
    collections::{BTreeSet, HashMap},
    fmt::Debug,
};

use itertools::Itertools;
use petgraph::unionfind::UnionFind;

use crate::{
    constraint::{
        tag::{ConstraintTag, Tag},
        ArityPredicate, ConditionalPredicate, EvaluatePredicate,
    },
    indexing::IndexKey,
    pattern::Satisfiable,
    Constraint,
};

use super::{ConstraintSelector, StringPatternPosition, StringSubjectPosition};

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

impl EvaluatePredicate<String, StringSubjectPosition> for CharacterPredicate {
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

/// Constraint tags for string pattern matching
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StringTag<K> {
    /// A constraint on a position of the input string
    Position(K),
}

impl<K: IndexKey> ConditionalPredicate<K> for CharacterPredicate {
    fn condition_on(
        &self,
        keys: &[K],
        known_constraints: &BTreeSet<Constraint<K, Self>>,
        prev_constraints: &[Constraint<K, Self>],
    ) -> Satisfiable<Constraint<K, Self>> {
        if prev_constraints
            .iter()
            .any(|c| c.predicate() == self && keys == c.required_bindings())
        {
            // Constraint must have been satisfied before
            return Satisfiable::No;
        }
        use CharacterPredicate::*;
        match self {
            BindingEq => {
                let (k0, k1) = (keys[0], keys[1]);
                simplify_binding_eq(k0, k1, known_constraints)
            }
            ConstVal(val) => simplify_const_val(*val, keys[0], known_constraints),
        }
    }
}

impl ConstraintTag<StringPatternPosition> for CharacterPredicate {
    type Tag = StringTag<StringPatternPosition>;

    fn get_tags(&self, keys: &[StringPatternPosition]) -> Vec<Self::Tag> {
        use CharacterPredicate::*;
        assert_eq!(self.arity(), keys.len());

        match self {
            BindingEq => {
                vec![StringTag::Position(keys[0]), StringTag::Position(keys[1])]
            }
            ConstVal(_) => vec![StringTag::Position(keys[0])],
        }
    }
}

impl Tag<StringPatternPosition, CharacterPredicate> for StringTag<StringPatternPosition> {
    type ExpansionFactor = ();

    type Evaluator = ConstraintSelector;

    fn expansion_factor<'c, C>(
        &self,
        _constraints: impl IntoIterator<Item = C>,
    ) -> Self::ExpansionFactor
    where
        C: Into<(&'c CharacterPredicate, &'c [StringPatternPosition])>,
    {
    }

    fn compile_evaluator<'c, C>(&self, constraints: impl IntoIterator<Item = C>) -> Self::Evaluator
    where
        C: Into<(&'c CharacterPredicate, &'c [StringPatternPosition])>,
    {
        let constraints = constraints.into_iter().map(|c| c.into()).collect_vec();
        ConstraintSelector::from_constraints(constraints)
    }
}

fn simplify_const_val<K: IndexKey>(
    val: char,
    k: K,
    known_constraints: &BTreeSet<Constraint<K, CharacterPredicate>>,
) -> Satisfiable<Constraint<K, CharacterPredicate>> {
    // Gather all ConstVal constraint
    let constvals = get_constvals(known_constraints);

    // If a ConstVal() already exists for `k`, then we must have `val == known_val`
    if let Some(&known_val) = constvals.get(&k) {
        if known_val == val {
            return Satisfiable::Tautology;
        } else {
            return Satisfiable::No;
        }
    }

    // We now consider BindingEqs: if it must be k == k2 and we have a constval
    // for k2, then we can deduce the value for k too.
    let equal_keys = find_equal_keys(k, known_constraints);

    // Same as above: if a ConstVal() already exists for some `k2`, then we must
    // have `val == known_val`
    for k2 in equal_keys {
        if let Some(&known_val) = constvals.get(&k2) {
            if known_val == val {
                return Satisfiable::Tautology;
            } else {
                return Satisfiable::No;
            }
        }
    }

    let self_constraint = Constraint::try_new(CharacterPredicate::ConstVal(val), vec![k]).unwrap();
    Satisfiable::Yes(self_constraint)
}

fn simplify_binding_eq<K: IndexKey>(
    k1: K,
    k2: K,
    known_constraints: &BTreeSet<Constraint<K, CharacterPredicate>>,
) -> Satisfiable<Constraint<K, CharacterPredicate>> {
    let equal_keys_1 = find_equal_keys(k1, known_constraints);

    if equal_keys_1.contains(&k2) {
        return Satisfiable::Tautology;
    }

    let equal_keys_2 = find_equal_keys(k2, known_constraints);
    let constvals = get_constvals(known_constraints);

    let val1 = equal_keys_1.iter().find_map(|k| constvals.get(k).copied());
    let val2 = equal_keys_2.iter().find_map(|k| constvals.get(k).copied());

    match (val1, val2) {
        (None, Some(val)) => {
            let constraint =
                Constraint::try_new(CharacterPredicate::ConstVal(val), vec![k1]).unwrap();
            Satisfiable::Yes(constraint)
        }
        (Some(val), None) => {
            let constraint =
                Constraint::try_new(CharacterPredicate::ConstVal(val), vec![k2]).unwrap();
            Satisfiable::Yes(constraint)
        }
        (Some(val1), Some(val2)) => {
            if val1 == val2 {
                Satisfiable::Tautology
            } else {
                Satisfiable::No
            }
        }
        (None, None) => {
            let constraint =
                Constraint::try_new(CharacterPredicate::BindingEq, vec![k1, k2]).unwrap();
            Satisfiable::Yes(constraint)
        }
    }
}

fn get_constvals<K: IndexKey>(
    known_constraints: &BTreeSet<Constraint<K, CharacterPredicate>>,
) -> HashMap<K, char> {
    let constvals = known_constraints
        .iter()
        .filter_map(|c| {
            let &CharacterPredicate::ConstVal(val) = c.predicate() else {
                return None;
            };
            (c.required_bindings()[0], val).into()
        })
        .into_group_map();
    constvals
        .into_iter()
        .map(|(k, vals)| {
            let v = vals[0];
            assert!(vals.iter().all(|&v2| v == v2));
            (k, v)
        })
        .collect()
}

fn find_equal_keys<K: IndexKey>(
    k: K,
    known_constraints: &BTreeSet<Constraint<K, CharacterPredicate>>,
) -> BTreeSet<K> {
    let binding_eqs = known_constraints
        .iter()
        .filter_map(|c| {
            let &CharacterPredicate::BindingEq = c.predicate() else {
                return None;
            };
            [c.required_bindings()[0], c.required_bindings()[1]].into()
        })
        .collect_vec();
    let mut all_keys: Vec<_> = binding_eqs.iter().flat_map(|&[k1, k2]| [k1, k2]).collect();
    all_keys.sort_unstable();
    all_keys.dedup();

    let Ok(pos1) = all_keys.binary_search(&k) else {
        return [k].into();
    };

    let mut uf = UnionFind::new(all_keys.len());

    for &[k1, k2] in &binding_eqs {
        let pos1 = all_keys.binary_search(&k1).unwrap();
        let pos2 = all_keys.binary_search(&k2).unwrap();
        uf.union(pos1, pos2);
    }

    all_keys
        .iter()
        .copied()
        .filter(|k| {
            let pos2 = all_keys.binary_search(k).unwrap();
            uf.find(pos1) == uf.find(pos2)
        })
        .collect()
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
        p.try_into_partial_pattern().unwrap();
    }
}
