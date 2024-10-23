use std::{fmt, iter};

use crate::string::CharVar;
use crate::{Constraint, HashMap, Pattern};

use super::{CharacterPredicate, MatrixPatternPosition};

/// A pattern for matching a matrix of characters.
///
/// A pattern is a sequence of rows, where each row is a sequence of characters.
/// Each character can be a literal or a variable.
/// A variable is denoted by a '$' character, followed by a single character.
#[derive(Clone)]
pub struct MatrixPattern {
    /// The rows of the matrix pattern.
    rows: Vec<Vec<Option<CharVar>>>,
}

impl MatrixPattern {
    /// Create a new matrix pattern from a sequence of rows.
    pub fn new(rows: Vec<Vec<Option<CharVar>>>) -> Self {
        Self { rows }
    }

    /// Parse a matrix pattern from a string.
    ///
    /// The matrix pattern is a sequence of lines of characters, one for each row.
    /// Each character in a row can be a literal or a variable.
    /// A variable is denoted by a '$' character, followed by a single character.
    ///
    /// Use a single dash to skip a character on a row. Spaces are ignored and
    /// can be used to align rows as desired.
    ///
    /// Example:
    /// ```text
    ///  a $b c
    ///  $b - d
    /// ```
    /// is parsed as `[["a", "$b", "c"], ["$b", (), "d"]]` where we used
    /// () to denote a missing character.
    pub fn parse_str(s: &str) -> Self {
        let rows = s.lines().map(Self::parse_row).collect::<Vec<_>>();

        Self { rows }
    }

    fn parse_row(row: &str) -> Vec<Option<CharVar>> {
        let mut char_iter = row.chars().filter(|c| !c.is_whitespace());
        let parsed_iter = iter::from_fn(|| match char_iter.next()? {
            '$' => Some(Some(CharVar::Variable(char_iter.next().unwrap()))),
            '-' => Some(None),
            c => Some(Some(CharVar::Literal(c))),
        });
        parsed_iter.collect()
    }

    fn enumerate(&self) -> impl Iterator<Item = (MatrixPatternPosition, CharVar)> + '_ {
        // We currently set the start at the top left of the pattern and only use
        // positive indices.
        self.rows.iter().enumerate().flat_map(|(i, row)| {
            row.iter().enumerate().filter_map(move |(j, &char_var)| {
                Some((MatrixPatternPosition(i as isize, j as isize), char_var?))
            })
        })
    }
}

impl Pattern for MatrixPattern {
    type Key = MatrixPatternPosition;
    type Predicate = CharacterPredicate;
    type Error = ();

    fn try_to_constraint_vec(
        &self,
    ) -> Result<Vec<Constraint<Self::Key, Self::Predicate>>, Self::Error> {
        // For a variable name, the first position it appears at
        let mut var_to_pos: HashMap<char, _> = Default::default();
        let mut constraints = Vec::new();
        for (index, char_var) in self.enumerate() {
            // Assign i-th variable
            match char_var {
                CharVar::Literal(c) => {
                    constraints.push(
                        Constraint::try_new(CharacterPredicate::ConstVal(c), vec![index]).unwrap(),
                    );
                }
                CharVar::Variable(c) => {
                    if let Some(&first_index) = var_to_pos.get(&c) {
                        constraints.push(
                            Constraint::try_new(
                                CharacterPredicate::BindingEq,
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
                Constraint::try_new(
                    CharacterPredicate::BindingEq,
                    vec![MatrixPatternPosition(0, 0), MatrixPatternPosition(0, 0)],
                )
                .unwrap(),
            );
        }
        Ok(constraints)
    }
}

impl fmt::Debug for MatrixPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\"\"\"")?;
        for row in &self.rows {
            for char_var in row {
                if let Some(char_var) = char_var {
                    write!(f, "{:?}", char_var)?;
                } else {
                    write!(f, "-")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "\"\"\"")
    }
}
