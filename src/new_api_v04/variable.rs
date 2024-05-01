//! Variables in constraints.
//!
//! Variable live in a symbol alphabet V. At runtime, they are bound to values
//! in the domain universe U.
//!
//! The bindings are stored and retrieve in a map-like struct that implements
//! the [VariableScope] trait.

use crate::HashMap;
use std::{fmt::Debug, hash::Hash};
use thiserror::Error;

/// Errors that occur when binding variables in scope.
#[derive(Debug, Clone, Error)]
pub enum BindVariableError {
    /// A variable already exists in the scope.
    #[error("Cannot bind variable {0}: already exists")]
    VariableExists(String),

    /// A value is already bound to another variable in scope.
    #[error("Cannot bind value {value} to variable {variable}: value already exists")]
    ValueExists {
        /// The value that already exists
        value: String,
        /// The variable binding the value to
        variable: String,
    },
}

/// A map-like trait for variable bindings.
pub trait VariableScope<V, U>: Clone {
    fn get(&self, var: &V) -> Option<&U>;
    fn bind(&mut self, var: V, val: U) -> Result<(), BindVariableError>;
}

impl<V: Eq + Hash + Debug + Clone, U: Clone> VariableScope<V, U> for HashMap<V, U> {
    fn get(&self, var: &V) -> Option<&U> {
        self.get(var)
    }

    fn bind(&mut self, var: V, val: U) -> Result<(), BindVariableError> {
        if self.contains_key(&var) {
            return Err(BindVariableError::VariableExists(format!("{:?}", var)));
        }
        self.insert(var, val);
        Ok(())
    }
}
