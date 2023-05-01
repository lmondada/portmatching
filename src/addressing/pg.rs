//! Addressing scheme for the `portgraph` crate.
use std::{iter::Cloned, slice::Iter};

use portgraph::{NodeIndex, PortGraph, PortOffset};

use crate::utils::{follow_path, iter::AddAsRefIt};

use super::{cache::AsSpineID, NodeOffset, Rib, SkeletonAddressing, SpineAddress};

/// Obtain portgraph paths from addresses.
pub trait AsPathOffset {
    /// The path from the root node to the addressed node.
    fn as_path_offset(&self) -> (&[PortOffset], usize);
}

impl AsPathOffset for (&[PortOffset], usize) {
    fn as_path_offset(&self) -> (&[PortOffset], usize) {
        (self.0, self.1)
    }
}

/// Addressing scheme for the `portgraph` crate.
///
/// Vertebrae are given by paths from the root node. Graph, spine and
/// ribs are stored as references.
///
/// If the spine is [`None`], the addressing scheme is not defined. If
/// the ribs are [`None`], any index is valid.
///
/// # Example
///
/// ```
/// use portgraph::{PortGraph, NodeIndex, PortOffset};
/// let mut g = PortGraph::new();
/// let n0 = g.add_node(0, 1);
/// let n1 = g.add_node(1, 2);
/// let n2 = g.add_node(1, 0);
/// let p0_out0 = g.port_index(n0, PortOffset::new_outgoing(0)).unwrap();
/// let p1_in0 = g.port_index(n1, PortOffset::new_incoming(0)).unwrap();
/// let p1_out1 = g.port_index(n1, PortOffset::new_outgoing(1)).unwrap();
/// let p2_in0 = g.port_index(n2, PortOffset::new_incoming(0)).unwrap();
/// g.link_ports(p0_out0, p1_in0).unwrap();
/// g.link_ports(p1_out1, p2_in0).unwrap();
///
/// let skel = Skeleton::new(&g, n0);
/// let addressing = PortGraphAddressing::from_skeleton(&skel);
/// assert_eq!(
///     addressing.get_addr(n2),
///     ((&[PortOffset::new_outgoing(0)], 1), 1)
/// );
/// ```
#[derive(Clone)]
pub struct PortGraphAddressing<'g, 'n, S> {
    root: NodeIndex,
    graph: &'g PortGraph,
    spine: Option<&'n Vec<S>>,
    ribs: Option<&'n Vec<Rib>>,
}

impl<'g, 'n, S> PortGraphAddressing<'g, 'n, S> {
    /// New addressing scheme relative to root.
    ///
    /// This scheme is useless unless at least the root and spine are set.
    pub fn new(
        root: NodeIndex,
        graph: &'g PortGraph,
        spine: Option<&'n Vec<S>>,
        ribs: Option<&'n Vec<Rib>>,
    ) -> Self {
        Self {
            root,
            graph,
            spine,
            ribs,
        }
    }
}

impl<'g, 'n, S: SpineAddress> PortGraphAddressing<'g, 'n, S> {
    /// Iterate over the spine and ribs.
    pub fn iter(&self) -> SkeletonIt<AddAsRefIt<Iter<'n, S>>, Cloned<Iter<'n, Rib>>> {
        SkeletonIt(SkeletonItEnum::new(
            self.spine.map(|spine| AddAsRefIt::new(spine.iter())),
            self.ribs.map(|ribs| ribs.iter().cloned()),
        ))
    }
}

impl<'g, 'n, S> PortGraphAddressing<'g, 'n, S> {
    /// Return a copy of self with a new spine.
    ///
    /// Also resets the ribs, as they are defined relative to the spine.
    pub fn with_spine(&self, spine: &'n Vec<S>) -> Self {
        PortGraphAddressing {
            root: self.root,
            graph: self.graph,
            spine: Some(spine),
            ribs: self.ribs,
        }
    }

    /// Return a copy of self with new ribs.
    ///
    /// Mostly useful to specify ribs when the addressing scheme has none.
    pub fn with_ribs(&self, ribs: &'n Vec<Rib>) -> Self {
        PortGraphAddressing {
            root: self.root,
            graph: self.graph,
            spine: self.spine,
            ribs: Some(ribs),
        }
    }

    /// The root of the addressing scheme.
    pub fn root(&self) -> NodeIndex {
        self.root
    }

    /// The port graph of the addressing scheme.
    pub fn graph(&self) -> &PortGraph {
        &self.graph
    }
}

/// Iterator over the spine and ribs for [`PortGraphAddressing`].
///
/// This is basically a zip iterator over the spine and ribs, but with
/// some extra logic to handle the case where the spine or ribs are
/// [`None`].
pub struct SkeletonIt<IS, IR>(SkeletonItEnum<IS, IR>);

enum SkeletonItEnum<IS, IR> {
    SomeSome(IS, IR),
    SomeNone(IS),
    One,
    None,
}

impl<IS, IR> SkeletonItEnum<IS, IR> {
    /// New iterator over the spine and ribs.
    pub fn new(spine: Option<IS>, ribs: Option<IR>) -> Self {
        match (spine, ribs) {
            (Some(spine), Some(ribs)) => Self::SomeSome(spine, ribs),
            (Some(spine), None) => Self::SomeNone(spine),
            (None, _) => Self::One,
        }
    }
}

impl<'s, IS, IR> Iterator for SkeletonIt<IS, IR>
where
    IS: Iterator + Clone,
    IR: Iterator + Clone,
    IS::Item: Default,
{
    type Item = (IS::Item, Option<IR::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            SkeletonItEnum::SomeSome(spine, ribs) => spine
                .next()
                .zip(ribs.next())
                .map(|(spine, ribs)| (spine, Some(ribs))),
            SkeletonItEnum::SomeNone(spine) => spine.next().map(|spine| (spine, None)),
            SkeletonItEnum::One => {
                *self = Self(SkeletonItEnum::None);
                Some((Default::default(), None))
            }
            SkeletonItEnum::None => None,
        }
    }
}

impl<'g, 'n, S> SkeletonAddressing<'g, 'n, S> for PortGraphAddressing<'g, 'n, S>
where
    S: SpineAddress,
    for<'m> S::AsRef<'m>: Copy + Default + AsPathOffset + AsSpineID,
{
    type SkeletonIt = SkeletonIt<AddAsRefIt<Iter<'n, S>>, Cloned<Iter<'n, Rib>>>;

    fn init(root: NodeIndex, graph: &'g PortGraph) -> Self {
        PortGraphAddressing::new(root, graph, None, None)
    }

    fn root(&self) -> NodeIndex {
        self.root
    }

    fn graph(&self) -> &'g PortGraph {
        &self.graph
    }

    fn skeleton_iter(&self) -> Self::SkeletonIt {
        self.iter()
    }

    fn compute_vertebra<A: AsSpineID + AsPathOffset>(&self, spine: A) -> Option<NodeOffset> {
        let (path, offset) = spine.as_path_offset();
        follow_path(path, self.root, self.graph()).map(|s| (s, offset))
    }

    fn with_spine(&self, spine: &'n Vec<S>) -> Self {
        self.with_spine(spine)
    }

    fn with_ribs(&self, ribs: &'n Vec<Rib>) -> Self {
        PortGraphAddressing {
            spine: self.spine,
            graph: self.graph,
            ribs: Some(ribs),
            root: self.root,
        }
    }
}

#[cfg(test)]
mod tests {
    use portgraph::{NodeIndex, PortGraph, PortOffset};
    use proptest::prelude::*;

    use crate::{
        addressing::PortGraphAddressing,
        addressing::{skeleton::Skeleton, SkeletonAddressing},
        utils::test_utils::gen_portgraph_connected,
    };
    use portgraph::proptest::gen_node_index;

    fn link(graph: &mut PortGraph, (out_n, out_p): (usize, usize), (in_n, in_p): (usize, usize)) {
        let out_n = NodeIndex::new(out_n);
        let in_n = NodeIndex::new(in_n);
        let out_p = graph
            .port_index(out_n, PortOffset::new_outgoing(out_p))
            .unwrap();
        let in_p = graph
            .port_index(in_n, PortOffset::new_incoming(in_p))
            .unwrap();
        graph.link_ports(out_p, in_p).unwrap();
    }

    #[test]
    fn a_simple_addr() {
        let mut g = PortGraph::new();
        g.add_node(2, 0);
        g.add_node(0, 2);
        link(&mut g, (1, 0), (0, 1));
        link(&mut g, (1, 1), (0, 0));
        let b = PortGraphAddressing::new(NodeIndex::new(0), &g, None, None);
        let spine = vec![(Vec::new(), 0), (Vec::new(), 1)];
        let ribs = vec![[-1, 0], [0, 0]];
        let b = b.with_spine(&spine).with_ribs(&ribs);
        let addr = b.get_addr(NodeIndex::new(1), &mut ()).unwrap();
        let root = (&[] as &[PortOffset], 0);
        assert_eq!(addr, (root, -1));
    }

    #[test]
    fn test_get_addr() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 3);
        g.add_node(1, 2);
        let n2 = g.add_node(2, 1);
        link(&mut g, (0, 0), (1, 0));
        link(&mut g, (1, 0), (2, 0));
        link(&mut g, (1, 1), (2, 1));
        g.add_node(1, 0);
        link(&mut g, (0, 1), (3, 0));
        let n4 = g.add_node(2, 2);
        let n5 = g.add_node(1, 0);
        let n6 = g.add_node(0, 1);
        link(&mut g, (4, 0), (0, 0));
        link(&mut g, (4, 1), (5, 0));
        link(&mut g, (6, 0), (4, 1));

        let skel = Skeleton::new(&g, NodeIndex::new(0));
        let addressing = PortGraphAddressing::from_skeleton(&skel);

        let root_addr = (&[] as &[PortOffset], 0);
        assert_eq!(addressing.get_addr(n0, &mut ()).unwrap(), (root_addr, 0));
        assert_eq!(addressing.get_addr(n2, &mut ()).unwrap(), (root_addr, 2));
        assert_eq!(addressing.get_addr(n4, &mut ()).unwrap(), (root_addr, -1));
        let addr = (&[PortOffset::new_incoming(0)] as &[PortOffset], 1);
        assert_eq!(addressing.get_addr(n5, &mut ()).unwrap(), (addr, 1));
        assert_eq!(addressing.get_addr(n6, &mut ()).unwrap(), (addr, -1));
    }

    #[test]
    fn test_get_addr_cylic() {
        let mut g = PortGraph::new();
        g.add_node(1, 1);
        let n1 = g.add_node(1, 1);
        let n2 = g.add_node(1, 1);
        link(&mut g, (0, 0), (1, 0));
        link(&mut g, (1, 0), (2, 0));
        link(&mut g, (2, 0), (0, 0));

        let skel = Skeleton::new(&g, NodeIndex::new(0));
        let addressing = PortGraphAddressing::from_skeleton(&skel);

        let ribs = vec![[0, 2]];
        let addressing = addressing.with_ribs(&ribs);
        let root_addr = (&[] as &[PortOffset], 0);
        assert_eq!(addressing.get_addr(n2, &mut ()).unwrap(), (root_addr, 2));

        let ribs = vec![[-2, 0]];
        let addressing = addressing.with_ribs(&ribs);
        assert_eq!(addressing.get_addr(n2, &mut ()).unwrap(), (root_addr, -1));
        assert_eq!(addressing.get_addr(n1, &mut ()).unwrap(), (root_addr, -2));
    }

    proptest! {
        #[test]
        fn prop_get_addr((g, n) in gen_node_index(gen_portgraph_connected(10, 4, 20))) {
            let skel = Skeleton::new(&g, NodeIndex::new(0));
            let addressing = PortGraphAddressing::from_skeleton(&skel);
            let addr = addressing.get_addr(n, &mut ()).unwrap();
            prop_assert_eq!(n, addressing.get_node(&addr, &mut ()).unwrap());
        }
    }
}
