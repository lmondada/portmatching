use std::{collections::hash_map::Entry, iter};

use crate::{predicate::PredicateLogic, Constraint, HashMap, Pattern};
use derive_more::Display;

use super::{predicate::CharacterPredicate, StringConstraint, StringPatternPosition};

/// A pattern for matching on strings.
///
/// A pattern is a sequence of elements, each of which is either a literal or a
/// variable (see [`CharVar`]).
///
/// For ease of parsing, both the value of literals and the variable names are
/// `char`s. At parsing time, variables are recognised by a '$' character,
/// followed by a single character.
#[derive(Clone)]
pub struct StringPattern(Vec<CharVar>);

/// A single element in a string pattern.
///
/// Always matches a single character in a string.
#[derive(Clone, Copy, Display)]
pub enum CharVar {
    /// A `char` character.
    #[display(fmt = "{}", _0)]
    Literal(char),
    /// A variable that matches any `char`, subject to constraints.
    ///
    /// Variable names are themselves `char`s. This is for ease of parsing: every
    /// character read when parsing is either a literal or a $, in which case the
    /// next character defines the variable name.
    #[display(fmt = "${}", _0)]
    Variable(char),
}

impl StringPattern {
    /// Create a new string pattern from a sequence of characters.
    pub fn new(chars: Vec<CharVar>) -> Self {
        Self(chars)
    }

    /// Parse a string pattern from a string.
    ///
    /// The string pattern is a sequence of characters, where each character can be a literal or a variable.
    /// A variable is denoted by a '$' character, followed by a single character.
    ///
    /// Examples:
    /// - "abc" -> [a, b, c]
    /// - "$a$b$c" -> [$a, $b, $c]
    ///
    /// Note that the `$` character cannot be parsed as a literal. It will always
    /// be interpreted as the start of a variable name.
    pub fn parse_str(s: &str) -> Self {
        let mut char_iter = s.chars();
        let parsed_iter = iter::from_fn(|| match char_iter.next()? {
            '$' => Some(CharVar::Variable(char_iter.next().unwrap())),
            c => Some(CharVar::Literal(c)),
        });
        Self(parsed_iter.collect())
    }

    fn enumerate(&self) -> impl Iterator<Item = (StringPatternPosition, CharVar)> + '_ {
        self.0
            .iter()
            .enumerate()
            .map(|(i, &char_var)| (StringPatternPosition(i), char_var))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Pattern for StringPattern {
    type Key = StringPatternPosition;
    type Logic = PredicateLogic<StringPatternPosition, CharacterPredicate>;

    type Constraint = StringConstraint;

    fn required_bindings(&self) -> Vec<Self::Key> {
        (0..self.len()).map(StringPatternPosition).collect()
    }

    fn into_logic(self) -> Self::Logic {
        // For a variable name, the first position it appears at
        let mut var_to_pos: HashMap<char, _> = Default::default();
        let mut constraints = Vec::new();
        for (index, char_var) in self.enumerate() {
            // Assign the index variable
            match char_var {
                CharVar::Literal(c) => {
                    constraints.push(
                        Constraint::try_new(CharacterPredicate::ConstVal(c), vec![index]).unwrap(),
                    );
                }
                CharVar::Variable(c) => match var_to_pos.entry(c) {
                    Entry::Occupied(first_index) => {
                        constraints.push(
                            Constraint::try_new(
                                CharacterPredicate::BindingEq,
                                vec![index, *first_index.get()],
                            )
                            .unwrap(),
                        );
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(index);
                    }
                },
            }
        }
        PredicateLogic::from_constraints(constraints)
    }
}

impl std::fmt::Debug for CharVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Debug for StringPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"")?;
        for char_var in &self.0 {
            write!(f, "{:?}", char_var)?;
        }
        write!(f, "\"")
    }
}
