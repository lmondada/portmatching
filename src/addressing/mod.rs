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

pub mod cache;
pub(crate) mod constraint;
pub mod pg;
mod skeleton;

pub use cache::CachedOption;
#[doc(inline)]
pub use pg::PortGraphAddressing;

pub(crate) use self::cache::{AddressCache, AsSpineID};
use self::pg::AsPathOffset;
pub use skeleton::Skeleton;

use crate::utils::port_opposite;
use ::portgraph::{NodeIndex, PortGraph};

/// A node address with vertebra ID and path index
pub(crate) type Address<SpineID> = (SpineID, isize);

/// The address of a vertebra.
///
/// Addresses on the spine are treated specially, as various formats can be
/// used that allow for caching and other tradeoffs. Other addresses are obtained
/// by tuples `(spine_addr, ind)`, where `ind` is the index on the rib defined
/// by the spine_addr.
///
/// Spine addresses specify a `NodeIndex` as well as an offset, thus identifying
/// a specific incoming/outgoing port at the node.
pub trait SpineAddress {
    /// The type of the address as a reference, for cheap copying.
    type AsRef<'n>
    where
        Self: 'n;

    /// Get the address as a reference.
    fn as_ref(&self) -> Self::AsRef<'_>;
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
/// on a common vertex `v` have the same port offset. In portgraph terms:
/// ```
/// # fn main() { foo(); }
/// # fn foo() -> Option<()> {
/// # use portgraph::PortGraph;
/// # let mut graph: PortGraph = PortGraph::new();
/// # let v = graph.add_node(1, 1);
/// # let (e1, e2) = (graph.input(v, 0)?, graph.output(v, 0)?);
/// assert_eq!(graph.port_offset(e1)?.index(), graph.port_offset(e2)?.index());
/// # Some(())
/// # }
/// ```
/// Given that the edges must be distinct, this implies that one edge is incoming
/// and the other outgoing in `v`.
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
/// [^trail]: We should strictly speaking be using the term "trail" here as we
/// allow repeated vertices in the path, but no repeated edges.
pub trait SkeletonAddressing<'g, 'n, T>: Sized
where
    T: SpineAddress + 'n,
    T::AsRef<'n>: Copy + AsSpineID + AsPathOffset,
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
    fn compute_vertebra<A: AsSpineID + AsPathOffset>(&self, spine: A) -> Option<NodeOffset>;

    /// The fixed root of the graph used as reference for addressing.
    fn root(&self) -> NodeIndex;

    /// The port graph of the addressing scheme.
    fn graph(&self) -> &'g PortGraph;

    /// Initialize an addressing scheme.
    fn init(root: NodeIndex, graph: &'g PortGraph) -> Self;

    /// Return a copy of self with new ribs.
    ///
    /// Mostly useful to specify ribs when the addressing scheme has none.
    fn with_ribs(&self, ribs: &'n [Rib]) -> Self;

    /// Return a copy of self with a new spine.
    ///
    /// Also resets the ribs, as they are defined relative to the spine.
    fn with_spine(&self, spine: &'n [T]) -> Self;

    /// Get the node index corresponding to an address.
    fn get_node<A: AsSpineID + AsPathOffset + Copy, C: AddressCache>(
        &self,
        addr: &Address<A>,
        cache: &mut C,
    ) -> Option<NodeIndex> {
        let &(spine_id, ind) = addr;
        // Let us first check if the value is in the cache
        let addr = spine_id.as_spine_id().map(|spine_id| (spine_id, ind));
        if let Some(cache) = addr.as_ref().map(|addr| cache.get_node(addr)) {
            if matches!(cache, CachedOption::None | CachedOption::Some(_)) {
                return cache.cached().expect("value in cache").map(|node| node.0);
            }
        }

        // Compute the value
        // 1. Obtain the node of the spine (if possible from the cache)
        let root_addr = spine_id.as_spine_id().map(|spine| (spine, 0));
        let mut rootoffset = None;
        if let Some(cache) = root_addr.as_ref().map(|addr| cache.get_node(addr)) {
            if matches!(cache, CachedOption::None | CachedOption::Some(_)) {
                rootoffset = cache.cached().expect("value in cache");
            }
        }
        let (root, offset) = rootoffset.or_else(|| self.compute_vertebra(spine_id))?;
        if let Some(root_addr) = root_addr {
            cache.set_node(&root_addr, (root, offset));
        }
        // 2. Find the starting in/out port
        let mut port = match ind {
            ind if ind > 0 => self.graph().output(root, offset),
            ind if ind < 0 => self.graph().input(root, offset),
            // must be 0, so the root is the node we are looking for
            _ => return Some(root),
        };
        // 3. Follow edges along `ind` links
        let mut node = self.graph().port_node(port?).expect("invalid port");
        let mut offset = self.graph().port_offset(port?).expect("invalid port");
        for _ in 0..ind.abs() {
            let next_port = self.graph().port_link(port?)?;
            node = self.graph().port_node(next_port).expect("invalid port");
            offset = self.graph().port_offset(next_port).expect("invalid port");
            port = port_opposite(next_port, self.graph());
        }
        if let Some(addr) = addr.as_ref() {
            cache.set_node(addr, (node, offset.index()));
        }
        Some(node)
    }

    /// Get the address corresponding to a node index.
    fn get_addr<C: AddressCache>(
        &self,
        node: NodeIndex,
        cache: &mut C,
    ) -> Option<Address<T::AsRef<'n>>> {
        for (spine, rib) in self.skeleton_iter() {
            let root_addr = spine.as_spine_id().map(|spine| (spine, 0));
            let mut rootoffset = None;
            if let Some(cache) = root_addr.as_ref().map(|addr| cache.get_node(addr)) {
                if matches!(cache, CachedOption::None | CachedOption::Some(_)) {
                    rootoffset = cache.cached().expect("value in cache");
                }
            }
            let Some((root, offset)) = rootoffset.or_else(|| self.compute_vertebra(spine)) else {
                continue;
            };
            let [bef, aft] = rib.unwrap_or([isize::MIN, isize::MAX]);
            let mut ind = 0;
            let mut port = self.graph().output(root, offset);
            let first_port = port;
            if root == node && ind >= bef && ind <= aft {
                return Some((spine, ind));
            }
            // Always prefer a positive ind over negative one
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
                if port == first_port {
                    // detected a cycle
                    break;
                }
            }
            port = self.graph().input(root, offset);
            let first_port = port;
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
                if port == first_port {
                    // detected a cycle
                    break;
                }
            }
        }
        None
    }
}
