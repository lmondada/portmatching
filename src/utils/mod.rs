//! Utility functions.

#[cfg(test)]
pub(crate) mod test;

pub(crate) mod collect_min;
mod connected_components;

pub use connected_components::{connected_components, is_connected};
