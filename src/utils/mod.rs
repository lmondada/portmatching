//! Utility functions.

#[cfg(test)]
pub(crate) mod test;

mod depth;
mod pre_order;
pub use depth::is_connected;
