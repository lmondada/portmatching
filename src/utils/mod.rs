//! Utility functions.

#[cfg(test)]
#[cfg(portgraph)]
pub(crate) mod test;

pub(crate) mod collect_min;
#[cfg(portgraph)]
mod connected_components;

#[cfg(portgraph)]
pub use connected_components::{connected_components, is_connected};
