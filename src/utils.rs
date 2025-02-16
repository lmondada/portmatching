//! Utility functions.

#[cfg(feature = "portgraph")]
pub(crate) mod portgraph;
#[cfg(all(feature = "portgraph", feature = "proptest"))]
pub mod test;

use itertools::Itertools;

#[cfg(feature = "portgraph")]
pub use portgraph::is_connected;
#[cfg(all(feature = "portgraph", feature = "proptest"))]
pub use test::gen_portgraph_connected;

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
