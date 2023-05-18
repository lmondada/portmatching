use std::{
    cmp,
    collections::BTreeMap,
    fmt::{self, Display},
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex};
use smallvec::SmallVec;

use super::{
    Constraint, ConstraintVec, ElementaryConstraint, NodeAddress, NodeRange, PortAddress,
    PortLabel, Skeleton, SpineAddress,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address {
    pub(super) addr: NodeAddress,
    pub(super) label: PortLabel,
}

type Graph<'g> = (&'g PortGraph, NodeIndex);
impl<'g> PortAddress<Graph<'g>> for Address {
    fn ports(&self, (g, root): Graph<'g>) -> Vec<PortIndex> {
        let Some(node) = self.addr.get_node(g, root) else { return vec![] };
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

/// Adjacency constraint for unweighted graphs.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnweightedAdjConstraint {
    Dangling,
    Link(ConstraintVec<()>),
}

impl UnweightedAdjConstraint {
    pub(crate) fn dangling() -> Self {
        Self::Dangling
    }

    pub(crate) fn link(
        label: PortLabel,
        address: NodeAddress,
        no_addresses: Vec<NodeRange>,
    ) -> Self {
        let constraints = no_addresses
            .into_iter()
            .map(ElementaryConstraint::NoMatch)
            .chain([ElementaryConstraint::PortLabel(label)])
            .chain([ElementaryConstraint::Match(address)])
            .collect();
        Self::Link(constraints)
    }
}

impl Constraint for UnweightedAdjConstraint {
    type Graph<'g> = (&'g PortGraph, NodeIndex);

    fn is_satisfied<'g, A>(&self, ports: &A, g: Self::Graph<'g>) -> bool
    where
        A: PortAddress<Self::Graph<'g>>,
    {
        let ports = ports.ports(g);
        match self {
            UnweightedAdjConstraint::Dangling => !ports.is_empty(),
            UnweightedAdjConstraint::Link(constraints) => ports
                .into_iter()
                .filter_map(|p| g.0.port_link(p))
                .any(|p| constraints.is_satisfied(p, g.0, g.1, &())),
        }
    }

    fn and(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Dangling, Self::Dangling) => Some(Self::Dangling),
            (Self::Dangling, Self::Link(_)) => Some(other.clone()),
            (Self::Link(_), Self::Dangling) => Some(self.clone()),
            (Self::Link(c1), Self::Link(c2)) => c1.and(c2).map(Self::Link),
        }
    }

    fn to_elementary(&self) -> Vec<Self>
    where
        Self: Clone,
    {
        let Self::Link(c) = self else {
            return vec![self.clone()];
        };
        let ConstraintVec::Vec(c) = c else {
            return vec![]
        };
        c.iter()
            .cloned()
            .map(|c| Self::Link(ConstraintVec::new(vec![c])))
            .collect()
    }
}

impl Display for UnweightedAdjConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dangling => write!(f, "dangling"),
            Self::Link(c) => write!(f, "{:?}", c),
        }
    }
}

impl<'g> Skeleton<'g> {
    pub(crate) fn get_coordinates(
        &self,
        port: PortIndex,
    ) -> (PortLabel, NodeAddress, Vec<NodeRange>) {
        let node = self.graph().port_node(port).expect("invalid pattern");
        let offset = self.graph().port_offset(port).expect("invalid pattern");
        let addr = self.get_node_addr(node);
        let label = match self.graph().port_direction(port).expect("invalid pattern") {
            Direction::Incoming => PortLabel::Incoming(offset.index()),
            Direction::Outgoing => PortLabel::Outgoing(offset.index()),
        };
        let no_addr = self.get_no_addresses(node);
        (label, addr, no_addr)
    }

    pub(crate) fn get_address(&self, port: PortIndex) -> Address {
        let (label, addr, _) = self.get_coordinates(port);
        Address { label, addr }
    }

    pub(crate) fn get_no_addresses(&self, node: NodeIndex) -> Vec<NodeRange> {
        if node == self.root {
            return vec![];
        }
        let spine_inst = self.instantiate_spine(&self.spine);
        let mut rev_inds: BTreeMap<_, Vec<_>> = Default::default();
        for (i, lp) in spine_inst.iter().enumerate() {
            if let Some(lp) = lp {
                rev_inds.entry(lp.line_ind).or_default().push(i);
            }
        }
        let mut all_addrs = Vec::new();
        for line in self.node2line[node.index()].iter() {
            for &spine_ind in rev_inds.get(&line.line_ind).unwrap_or(&Vec::new()) {
                let spine = spine_inst[spine_ind].expect("By construction of in rev_inds");
                let ind = line.ind - spine.ind;
                all_addrs.push((spine_ind, ind))
            }
        }
        // Lower spine indices come first, prioritising positive indices
        all_addrs.sort_unstable_by_key(|addr| (addr.0, addr.1 < 0, addr.1.abs()));
        let addr = all_addrs
            .into_iter()
            .next()
            .expect("must have at least one address");
        let mut ribs = self.get_ribs(&self.spine);
        let mut spine = self.spine.clone();
        match addr.1.cmp(&0) {
            cmp::Ordering::Greater => {
                spine.truncate(addr.0 + 1);
                ribs.truncate(addr.0 + 1);
                ribs[addr.0] = (0..=addr.1 - 1).try_into().unwrap();
            }
            cmp::Ordering::Less => {
                spine.truncate(addr.0 + 1);
                ribs.truncate(addr.0 + 1);
                ribs[addr.0] = (addr.1 + 1..=ribs[addr.0].end()).try_into().unwrap();
            }
            cmp::Ordering::Equal => {
                spine.truncate(addr.0);
                ribs.truncate(addr.0);
            }
        }
        spine
            .into_iter()
            .zip(ribs)
            .map(|(spine, range)| NodeRange { spine, range })
            .collect()
    }

    pub(crate) fn get_node_addr(&self, node: NodeIndex) -> NodeAddress {
        if node == self.root {
            return NodeAddress {
                spine: SpineAddress {
                    path: SmallVec::new(),
                    offset: 0,
                },
                ind: 0,
            };
        }
        let spine_inst = self.instantiate_spine(&self.spine);
        let mut rev_inds: BTreeMap<_, Vec<_>> = Default::default();
        for (i, lp) in spine_inst.iter().enumerate() {
            if let Some(lp) = lp {
                rev_inds.entry(lp.line_ind).or_default().push(i);
            }
        }
        let mut all_addrs = Vec::new();
        for line in self.node2line[node.index()].iter() {
            for &spine_ind in rev_inds.get(&line.line_ind).unwrap_or(&Vec::new()) {
                let spine = spine_inst[spine_ind].expect("By construction of in rev_inds");
                let ind = line.ind - spine.ind;
                all_addrs.push((spine_ind, ind))
            }
        }
        // Lower spine indices come first, prioritising positive indices
        all_addrs.sort_unstable_by_key(|addr| (addr.0, addr.1 < 0, addr.1.abs()));
        let addr = all_addrs
            .into_iter()
            .next()
            .expect("must have at least one address");
        NodeAddress {
            spine: self.spine[addr.0].clone(),
            ind: addr.1,
        }
    }
}
