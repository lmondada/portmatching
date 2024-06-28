use std::iter;

use crate::{HashMap, Pattern};

use super::{constraint::StringConstraint, predicate::StringPredicate, StringIndexKey};

/// A pattern for matching a string.
///
/// A pattern is a sequence of characters, where each character can be a literal or a variable.
/// A variable is denoted by a '$' character, followed by a single character.
#[derive(Clone)]
pub struct StringPattern(Vec<CharVar>);

/// A character in a string pattern.
#[derive(Clone, Copy)]
pub enum CharVar {
    /// A `char` character.
    Literal(char),
    /// A character expressed by a variable.
    ///
    /// Variable names are themselves `char`s.
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
    pub fn parse_str(s: &str) -> Self {
        let mut char_iter = s.chars();
        let parsed_iter = iter::from_fn(|| match char_iter.next()? {
            '$' => Some(CharVar::Variable(char_iter.next().unwrap())),
            c => Some(CharVar::Literal(c)),
        });
        Self(parsed_iter.collect())
    }

    fn enumerate(&self) -> impl Iterator<Item = (StringIndexKey, CharVar)> + '_ {
        self.0
            .iter()
            .enumerate()
            .map(|(i, &char_var)| (StringIndexKey(i), char_var))
    }
}

impl Pattern for StringPattern {
    type Constraint = StringConstraint;

    fn to_constraint_vec(&self) -> Vec<Self::Constraint> {
        // For a variable name, the first position it appears at
        let mut var_to_pos: HashMap<char, _> = Default::default();
        let mut constraints = Vec::new();
        for (index, char_var) in self.enumerate() {
            // Assign the index variable
            match char_var {
                CharVar::Literal(c) => {
                    constraints.push(
                        Self::Constraint::try_new(StringPredicate::ConstVal(c), vec![index])
                            .unwrap(),
                    );
                }
                CharVar::Variable(c) => {
                    if let Some(&first_index) = var_to_pos.get(&c) {
                        constraints.push(
                            Self::Constraint::try_new(
                                StringPredicate::BindingEq,
                                vec![index, first_index],
                            )
                            .unwrap(),
                        );
                    } else {
                        var_to_pos.insert(c, index);
                    }
                }
            }
        }
        if constraints.is_empty() {
            // We add one (dummy) constraint for the empty pattern, forcing
            // the matcher to bind the first character to a position in the
            // string when matched. An alternative would be to explicitly
            // disallow empty patterns.
            constraints.push(
                Self::Constraint::try_new(
                    StringPredicate::BindingEq,
                    vec![StringIndexKey(0), StringIndexKey(0)],
                )
                .unwrap(),
            );
        }
        constraints
    }
}

impl std::fmt::Debug for CharVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CharVar::Literal(c) => write!(f, "{}", c),
            CharVar::Variable(c) => write!(f, "${}", c),
        }
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
