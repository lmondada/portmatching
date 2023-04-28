//! Addressing schemes for port graphs.
//!
//! An address is a unique node index that is invariant under graph isomorphisms
//! and graph embeddings (monomorphisms).
//! More precisely, for any [`PortGraph`] there is a bijection between [`NodeIndex`]s
//! and addresses that is preserved under isomorphisms.
//!
//! These are typically defined relative to a root by a path from root to node.
//! However, paths in graphs are not necessarily unique. Furthermore, even unique paths
//! might no longer be unique once embedded in a larger graph. To avoid these issues,
//! path addresses must satisfy the following conditions:
//!  * Paths must be canonical -- for every node there is a unique path from
//!    the root to it that is canonical.
//!  * Paths must be contained within a given induced subgraph -- think of it
//!    as a list of "allowed" nodes that can be traversed by the path
//! The second condition ensures that if a path is unique, then by keeping the
//! induced subgraph unchanged, it will still be unique once embedded within a
//! larger graph.

pub mod portgraph;
pub mod cache;
mod skeleton;

#[doc(inline)]
pub use self::portgraph::PortGraphAddressing;
pub use cache::CachedOption;

pub(crate) use skeleton::Skeleton;

use self::cache::{AddressCache, AsSpineID};
use ::portgraph::{NodeIndex, PortGraph};
use crate::utils::port_opposite;

/// A node address with vertebra ID and path index
pub(crate) type Address<SpineID> = (SpineID, isize);

pub(crate) trait SpineAddress {
    type AsRef<'n>
    where
        Self: 'n;

    fn as_ref<'n>(&'n self) -> Self::AsRef<'n>;
}

/// Spine: a list of vertebrae and their offsets
pub(crate) type Spine<T> = Vec<T>;
/// The endpoints of a rib, given as path indices
pub(crate) type Rib = [isize; 2];

/// An address, but specific to a given graph
type NodeOffset = (NodeIndex, usize);

/// An Addressing scheme based on graph skeletons.
///
/// This trait uses skeletons of graphs to define unique addresses. Skeletons
/// are defined by a spine, made of vertebrae, and (optionally) of ribs.
/// There are always as many vertebrae as there are ribs.
/// You can optionally provide an [`AddressCache`] to cache address computation.
/// 
/// An addressing scheme is specific to a graph (with lifetime `'g`) and spines
/// and ribs (with lifetime `'n`). These are stored as references.
/// Addresses are composed of a vertebra and an index along its rib. As vertebrae
/// are typically defined by a path from a fixed root vertex, they can be expensive
/// to copy, which is what the copyable reference type [`SpineAddress::AsRef`] is for.
/// 
/// In principle, this trait would not have to be specific to [`PortGraph`]s, as
/// the actual graph data structure could be abstracted away with only the addresses
/// being exposed to the graph trie. This would require a (welcome) refactor.
///
/// # Introduction
///
/// Consider paths[^trail] such that any two successive edges `e1` and `e2` incident
/// on a common vertex `v` have the same port offset. In pseudo-porgraph:
/// ```ignore
/// graph.port_offset(e1).index() == graph.port_offset(e2).index()
/// ```
/// Given that the edges must be distinct, this implies that one edge is incoming
/// and the other outgoing.
///
/// ## Skeleton paths
///
/// Paths as above are uniquely defined by any one of their port -- or rather, any such
/// path can be uniquely extended to a maximal such path.
/// We can consider a minimal set of ports such that any node of the graph is contained
/// in at least one of the maximal paths.
///
/// The union of these maximal paths is called the *skeleton* of the graph.
/// Continuing the analogy, we call the minimal set of ports that define the skeleton
/// the *spine*, each port is a *vertebra* and each maximal path is a *rib*.
///
/// Under these definitions, any node can be uniquely identified by a vertebra
/// and an index, corresponding to the length of the path along the rib. We use
/// the convention that the index is positive when following the directedness of the
/// rib and negative otherwise.
///
/// ## Graph regions
///
/// When the graph is fixed, vertebrae are given by a pair `(NodeIndex, offset)`,
/// where `offset` is a `usize` corresponding to the port offset. The graph-invariant
/// representation replaces `NodeIndex` with a vertebra index `VIx` that can be
/// translated into a `NodeIndex` for any graph using [`self.compute_vertebra`].
///
/// Addresses are defined relative to:
///  * a spine, i.e. a list of vertebrae
///  * rib intervals, indicating for each rib the minimum and maximum index allowed
///
/// [trail]: We should strictly speaking be using the term "trail" here as we
/// allow repeated vertices in the path, but no repeated edges.
pub trait SkeletonAddressing<'g, 'n, T: SpineAddress + 'n>: Sized
where
    T::AsRef<'n>: Copy + AsSpineID,
{
    /// Iterator over the pairs (vertebra, rib) of the skeleton.
    ///
    /// The type returned by [`Self::skeleton_iter`].
    type SkeletonIt: Iterator<Item = (T::AsRef<'n>, Option<Rib>)>;

    /// Iterator over the skeleton of the graph.
    ///
    /// The iterator yields pairs of a vertebra and an optional rib, given as an
    /// interval of minimum and maximum indices (included). If no rib is provided,
    /// then any rib index is valid.
    fn skeleton_iter(&self) -> Self::SkeletonIt;

    /// Map a vertebra index to a node index.
    ///
    /// This function is used to translate a vertebra index into a node index.
    /// In case caching is used, these calls will be cached and only computed
    /// when no cached entry is found.
    fn compute_vertebra(&'n self, spine: T::AsRef<'n>) -> Option<NodeOffset>;

    /// The fixed root of the graph used as reference for addressing.
    fn root(&self) -> NodeIndex;

    /// The port graph of the addressing scheme.
    fn graph(&self) -> &'g PortGraph;

    /// Initialize an addressing scheme.
    fn init(root: NodeIndex, graph: &'g PortGraph) -> Self;

    /// Return a copy of self with new ribs.
    /// 
    /// Mostly useful to specify ribs when the addressing scheme has none.
    fn with_ribs(&self, ribs: &'n Vec<Rib>) -> Self;

    /// Return a copy of self with a new spine.
    /// 
    /// Also resets the ribs, as they are defined relative to the spine.
    fn with_spine(&self, spine: &'n Vec<T>) -> Self;

    /// Get the node index corresponding to an address.
    fn get_node<C: AddressCache>(
        &'n self,
        addr: &Address<T::AsRef<'n>>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let &(spine_id, ind) = addr;
        cache
            .get_node(addr)
            .cached()?
            .map(|addr| addr.0)
            .or_else(|| {
                let (root, offset) = cache
                    .get_node(&(spine_id.as_spine_id(), 0))
                    .cached()?
                    .or_else(|| self.compute_vertebra(spine_id))?;
                if ind == 0 {
                    cache.set_node(addr, root);
                    return Some(root);
                }
                let mut port = match ind {
                    ind if ind > 0 => self.graph().output(root, offset),
                    ind if ind < 0 => self.graph().input(root, offset),
                    _ => self
                        .graph()
                        .output(root, offset)
                        .or(self.graph().input(root, offset)),
                };
                let mut node = self.graph().port_node(port?).expect("invalid port");
                for _ in 0..ind.abs() {
                    let next_port = self.graph().port_link(port?)?;
                    node = self.graph().port_node(next_port).expect("invalid port");
                    port = port_opposite(next_port, self.graph());
                }
                cache.set_node((spine_id.as_spine_id(), ind), node);
                Some(node)
            })
    }

    /// Get the address corresponding to a node index.
    fn get_addr<C: AddressCache>(&'n self, node: NodeIndex, cache: &mut C) -> Option<Address<T::AsRef<'n>>> {
        for (spine, rib) in self.skeleton_iter() {
            if let Some((root, offset)) = cache
                .get_node(&(spine, 0))
                .cached()?
                .or_else(|| self.compute_vertebra(spine))
            {
                let [bef, aft] = rib.unwrap_or([isize::MIN, isize::MAX]);
                let mut ind = 0;
                let mut port = self.graph().output(root, offset);
                if root == node && ind >= bef && ind <= aft {
                    return Some((spine, ind));
                }
                while port.is_some() && ind < aft {
                    port = self.graph().port_link(port.unwrap());
                    ind += 1;
                    if let Some(port_some) = port {
                        let curr_node = self.graph().port_node(port_some).expect("invalid port");
                        if curr_node == node && ind >= bef {
                            return Some((spine, ind));
                        }
                        port = port_opposite(port_some, self.graph());
                    }
                }
                port = self.graph().input(root, offset);
                ind = 0;
                while port.is_some() && ind > bef {
                    port = self.graph().port_link(port.unwrap());
                    ind -= 1;
                    if let Some(port_some) = port {
                        let curr_node = self.graph().port_node(port_some).expect("invalid port");
                        if curr_node == node && ind <= aft {
                            return Some((spine, ind));
                        }
                        port = port_opposite(port_some, self.graph());
                    }
                }
            }
        }
        None
    }
}
