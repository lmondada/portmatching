use std::iter::repeat;

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

mod elementary;
mod skeleton;
pub mod unweighted;
mod vec;
pub mod weighted;

use elementary::{ElementaryConstraint, PortLabel};

pub use skeleton::Skeleton;
use smallvec::SmallVec;
pub(crate) use unweighted::UnweightedAdjConstraint;
pub(crate) use weighted::WeightedAdjConstraint;

use crate::utils::{follow_path, port_opposite, ZeroRange};

pub(self) use vec::ConstraintVec;

/// Constraints for graph trie transitions.
///
/// To each transition in a graph trie corresponds a constraint. If a constraint
/// is satisfied, then the transition is valid.
///
/// The simplest constraint is the [`UnweightedConstraint`], which only checks
/// for adjacency. More complex constraints can be defined by implementing this
/// trait, for example for weighted or hierarchical graphs.
pub trait Constraint
where
    Self: Sized,
{
    /// The type of the graph the constraint is applied to.
    type Graph<'g>;

    /// Check if the constraint is satisfied.
    fn is_satisfied<'g, A>(&self, ports: &A, g: Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>;

    /// Merge two constraints.
    fn and(&self, other: &Self) -> Option<Self>;

    /// Express the constraint as multiple elementary constraints.
    fn to_elementary(&self) -> Vec<Self>
    where
        Self: Clone,
    {
        vec![self.clone()]
    }
}

pub trait PortAddress<Graph>: Clone + PartialEq + Eq + PartialOrd + Ord {
    fn ports(&self, g: Graph) -> Vec<PortIndex>;

    fn port(&self, g: Graph) -> Option<PortIndex> {
        let ports = self.ports(g);
        (ports.len() == 1).then_some(ports[0])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpineAddress {
    path: SmallVec<[PortOffset; 4]>,
    offset: usize,
}

impl SpineAddress {
    /// Node at the address
    fn get_node(&self, g: &PortGraph, root: NodeIndex) -> Option<NodeIndex> {
        follow_path(&self.path, root, g)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeAddress {
    spine: SpineAddress,
    ind: isize,
}

impl NodeAddress {
    /// Node at the address
    fn get_node(&self, g: &PortGraph, root: NodeIndex) -> Option<NodeIndex> {
        let root = self.spine.get_node(g, root)?;

        if self.ind == 0 {
            return Some(root);
        }
        let mut port = if self.ind < 0 {
            g.input(root, self.spine.offset)
        } else {
            g.output(root, self.spine.offset)
        };
        let mut node = g.port_node(port?).expect("invalid port");
        for _ in 0..self.ind.abs() {
            port = g.port_link(port?);
            node = g.port_node(port?).expect("invalid port");
            port = port_opposite(port?, g);
        }
        Some(node)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeRange {
    spine: SpineAddress,
    range: ZeroRange,
}

impl NodeRange {
    fn contains(&self, node: &NodeAddress) -> bool {
        self.spine == node.spine && self.range.contains(node.ind)
    }

    /// Whether node does not appear in the address range
    fn verify_no_match(&self, node: NodeIndex, g: &PortGraph, root: NodeIndex) -> bool {
        let Self { spine, range } = self;

        let Some(root) = follow_path(&spine.path, root, g) else {
            return true
        };
        if root == node {
            return false;
        }

        let n_neg = -range.start() as usize;
        let n_pos = if range.end() >= 0 {
            range.end() as usize
        } else {
            0
        };

        // go in both directions from root
        for (port, n_jumps) in [
            (g.output(root, spine.offset), n_pos),
            (g.input(root, spine.offset), n_neg),
        ] {
            let Some(port) = port else { continue };
            if n_times(n_jumps)
                .scan(port, |port, ()| {
                    let next_port = g.port_link(*port)?;
                    let node = g.port_node(next_port).expect("invalid port");
                    *port = port_opposite(next_port, g)?;
                    Some(node)
                })
                .any(|in_range| node == in_range)
            {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address {
    addr: NodeAddress,
    label: PortLabel,
    no_addr: Vec<NodeRange>,
}

type Graph<'g> = (&'g PortGraph, NodeIndex);
impl<'g> PortAddress<Graph<'g>> for Address {
    fn ports(&self, (g, root): Graph<'g>) -> Vec<PortIndex> {
        let Some(node) = self.addr.get_node(g, root) else { return vec![] };
        if self
            .no_addr
            .iter()
            .any(|range| !range.verify_no_match(node, g, root))
        {
            return vec![];
        }
        let as_vec = |p: Option<_>| p.into_iter().collect();
        match self.label {
            PortLabel::Outgoing(out_p) => as_vec(g.output(node, out_p)),
            PortLabel::Incoming(in_p) => as_vec(g.input(node, in_p)),
        }
    }
}

type GraphBis<'g, V> = (&'g PortGraph, V, NodeIndex);
impl<'g, V> PortAddress<GraphBis<'g, V>> for Address {
    fn ports(&self, (g, _, root): GraphBis<'g, V>) -> Vec<PortIndex> {
        let Some(node) = self.addr.get_node(g, root) else { return vec![] };
        let as_vec = |p: Option<_>| p.into_iter().collect();
        match self.label {
            PortLabel::Outgoing(out_p) => as_vec(g.output(node, out_p)),
            PortLabel::Incoming(in_p) => as_vec(g.input(node, in_p)),
        }
    }
}

fn n_times(n: usize) -> impl Iterator<Item = ()> {
    repeat(()).take(n)
}
