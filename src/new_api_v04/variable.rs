//! Variables in constraints.
//!
//! Variable live in a symbol alphabet V. At runtime, they are bound to values
//! in the domain universe U.
//!
//! The bindings are stored and retrieve in a map-like struct that implements
//! the [VariableScope] trait.

/// Errors that occur when binding variables in scope.
#[derive(Debug, Clone)]
pub enum BindVariableError {
    /// A variable already exists in the scope.
    VariableExists,
    /// A value is already bound to another variable in scope.
    ValueExists,
}

/// A map-like trait for variable bindings.
pub trait VariableScope<V, U> {
    fn get(&self, var: &V) -> Option<&U>;
    fn bind(&mut self, var: &V, val: U) -> Result<(), BindVariableError>;
}
