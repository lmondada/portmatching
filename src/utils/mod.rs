mod pre_order;
mod depth;

pub(crate) use depth::{centre, NoCentreError};

#[cfg(test)]
pub(crate) mod test_utils;