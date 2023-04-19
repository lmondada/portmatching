pub(crate) mod address;
pub(crate) mod cover;
mod depth;
mod pre_order;

pub use depth::is_connected;
pub(crate) use depth::{centre, NoCentreError};

#[cfg(test)]
pub(crate) mod test_utils;
