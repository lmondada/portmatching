//! Utility functions.

pub(crate) mod mark_last;
// #[cfg(feature = "portgraph")]
// pub(crate) mod portgraph;
// #[cfg(all(feature = "portgraph", feature = "proptest"))]
// pub mod test;
mod toposort;
pub(crate) mod tracer;

use itertools::Itertools;
pub(crate) use mark_last::mark_last;
// #[cfg(feature = "portgraph")]
// pub(crate) use portgraph::{connected_components, is_connected};
// #[cfg(all(feature = "portgraph", feature = "proptest"))]
// pub use test::gen_portgraph_connected;
pub(crate) use toposort::{online_toposort, OnlineToposort};
pub(crate) use tracer::{TracedNode, Tracer};

/// Sort a vector and return a vector of pairs of the original value and its position.
#[allow(dead_code)]
pub(crate) fn sort_with_indices<V: Ord>(vec: impl IntoIterator<Item = V>) -> Vec<(V, usize)> {
    let mut vec_inds = vec
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect_vec();
    vec_inds.sort_by(|(c1, _), (c2, _)| c1.cmp(c2));
    vec_inds
}
