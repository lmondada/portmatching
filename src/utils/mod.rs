//! Utility functions.

// #[cfg(test)]
// #[cfg(feature = "portgraph")]
// pub(crate) mod test;

// #[cfg(feature = "portgraph")]
// pub(crate) mod collect_min;
#[cfg(feature = "portgraph")]
mod connected_components;

#[cfg(feature = "portgraph")]
pub use connected_components::{connected_components, is_connected};
