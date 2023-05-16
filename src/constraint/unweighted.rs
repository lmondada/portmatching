use std::{
    cmp,
    collections::{BTreeMap, BTreeSet},
    fmt::Display,
    mem,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::utils::{follow_path, port_opposite};

use super::{Constraint, PortAddress, Skeleton};

// TODO: use Rc or other for faster speed
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NodeAddress {
    pub(super) no_match: Vec<(Vec<PortOffset>, usize, [isize; 2])>,
    pub(super) the_match: (Vec<PortOffset>, usize, isize),
}

fn find_match(
    &(ref spine, offset, ind): &(Vec<PortOffset>, usize, isize),
    g: &PortGraph,
    root: NodeIndex,
) -> Option<NodeIndex> {
    let root = follow_path(spine, root, g)?;
    if ind == 0 {
        return Some(root);
    }
    let mut port = if ind < 0 {
        g.input(root, offset)
    } else {
        g.output(root, offset)
    };
    let mut node = g.port_node(port?).expect("invalid port");
    for _ in 0..ind.abs() {
        port = g.port_link(port?);
        node = g.port_node(port?).expect("invalid port");
        port = port_opposite(port?, g);
    }
    Some(node)
}

// TODO: maybe use skeleton
fn verify_no_match(
    no_match: &[(Vec<PortOffset>, usize, [isize; 2])],
    node: NodeIndex,
    g: &PortGraph,
    root: NodeIndex,
) -> bool {
    for &(ref spine, offset, [bef, aft]) in no_match.iter() {
        let Some(root) = follow_path(spine, root, g) else {
            continue;
        };
        if root == node {
            return false;
        }
        let mut port = g.output(root, offset);
        // Loop over positive inds
        for _ in 0..aft {
            if port.is_none() {
                break;
            }
            port = g.port_link(port.unwrap());
            if let Some(port_some) = port {
                let curr_node = g.port_node(port_some).expect("invalid port");
                if curr_node == node {
                    return false;
                }
                port = port_opposite(port_some, g);
            }
        }
        port = g.input(root, offset);
        // Loop over negative inds
        for _ in ((bef + 1)..=0).rev() {
            if port.is_none() {
                break;
            }
            port = g.port_link(port.unwrap());
            if let Some(port_some) = port {
                let curr_node = g.port_node(port_some).expect("invalid port");
                if curr_node == node {
                    return false;
                }
                port = port_opposite(port_some, g);
            }
        }
    }
    true
}

impl NodeAddress {
    pub fn node(&self, g: &PortGraph, root: NodeIndex) -> Option<NodeIndex> {
        let node = find_match(&self.the_match, g, root)?;

        // Check there is no address that also points to node in `no_match`
        verify_no_match(&self.no_match, node, g, root).then_some(node)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PortLabel {
    Outgoing(usize),
    Incoming(usize),
}
impl PortLabel {
    fn and(self, e: PortLabel) -> Option<PortLabel> {
        if self == e {
            Some(self)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address {
    pub(super) addr: NodeAddress,
    pub(super) label: PortLabel,
}

impl PortAddress for Address {
    fn ports(&self, g: &PortGraph, root: NodeIndex) -> Vec<PortIndex> {
        let Some(node) = self.addr.node(g, root) else { return vec![] };
        let as_vec = |p: Option<_>| p.into_iter().collect();
        match self.label {
            PortLabel::Outgoing(out_p) => as_vec(g.output(node, out_p)),
            PortLabel::Incoming(in_p) => as_vec(g.input(node, in_p)),
        }
    }
}

/// A state transition for an unweighted graph trie.
///
/// This corresponds to following an edge of the input graph.
/// This edge is given by one of the outgoing port at the current node.
/// Either the port exists and is connected to another port, or the port exist
/// but is unlinked (it is "dangling"), or the port does not exist.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnweightedConstraint {
    // All constraints must be satisfied
    AllAdjacencies {
        label: PortLabel,
        no_match: Vec<(Vec<PortOffset>, usize, [isize; 2])>,
        the_matches: Vec<(Vec<PortOffset>, usize, isize)>,
    },
    // Port must be linked to one of `other_ports`
    Adjacency {
        other_ports: Address,
    },
    // Port must be dangling (at least existing)
    Dangling,
}

impl UnweightedConstraint {
    fn to_vec(&self) -> Vec<Self> {
        let UnweightedConstraint::AllAdjacencies {
            label,
            no_match,
            the_matches,
        } = self else { return vec![self.clone()] };
        let mut no_match = no_match.clone();
        the_matches
            .iter()
            .map(|m| UnweightedConstraint::Adjacency {
                other_ports: Address {
                    addr: NodeAddress {
                        no_match: mem::take(&mut no_match),
                        the_match: m.clone(),
                    },
                    label: *label,
                },
            })
            .collect()
    }
}

impl Constraint for UnweightedConstraint {
    type Address = Address;

    fn is_satisfied(&self, this_ports: &Address, g: &PortGraph, root: NodeIndex) -> bool {
        match self {
            UnweightedConstraint::Adjacency { other_ports } => {
                let other_ports = other_ports.ports(g, root);
                let this_ports = this_ports.ports(g, root);
                adjacency_constraint(g, this_ports, other_ports)
            }
            UnweightedConstraint::Dangling => !this_ports.ports(g, root).is_empty(),
            UnweightedConstraint::AllAdjacencies { .. } => self
                .to_vec()
                .iter()
                .all(|c| c.is_satisfied(this_ports, g, root)),
        }
    }

    // Merge two constraints
    fn and(&self, other: &UnweightedConstraint) -> Option<UnweightedConstraint> {
        // Gather all constraints in a vec
        simplify_constraints(vec![self.clone(), other.clone()])
    }
}

fn simplify_constraints(constraints: Vec<UnweightedConstraint>) -> Option<UnweightedConstraint> {
    let constraints = flatten_constraints(constraints);

    // If we only have dangling, then dangling. Otherwise we can remove danglings
    if constraints
        .iter()
        .all(|c| c == &UnweightedConstraint::Dangling)
    {
        return Some(UnweightedConstraint::Dangling);
    }

    let mut matches = BTreeSet::new();
    let mut no_matches = BTreeSet::new();
    let mut all_labels = Vec::new();
    for c in constraints {
        if let UnweightedConstraint::Adjacency { other_ports: ports } = c {
            let node = ports.addr;
            let addr = node.the_match;
            matches.insert(addr);
            no_matches.extend(node.no_match.into_iter());
            all_labels.push(ports.label);
        }
    }

    let Some(label) = all_labels
        .into_iter()
        .fold(None, |acc: Option<PortLabel>, e| {
            acc.map(|acc| acc.and(e)).unwrap_or(Some(e))
        })
    else { return None };

    for &(ref path, offset, ind) in matches.iter() {
        if no_matches
            .iter()
            .any(|&(ref no_path, no_offset, [bef, aft])| {
                no_path == path && no_offset == offset && (bef..=aft).contains(&ind)
            })
        {
            return None;
        }
    }

    match matches.len() {
        0 => None,
        1 => Some(UnweightedConstraint::Adjacency {
            other_ports: Address {
                addr: NodeAddress {
                    no_match: no_matches.into_iter().collect(),
                    the_match: matches.into_iter().next().unwrap(),
                },
                label,
            },
        }),
        _ => Some(UnweightedConstraint::AllAdjacencies {
            label,
            no_match: no_matches.into_iter().collect(),
            the_matches: matches.into_iter().collect(),
        }),
    }
}

fn flatten_constraints(constraints: Vec<UnweightedConstraint>) -> Vec<UnweightedConstraint> {
    constraints.into_iter().flat_map(|c| c.to_vec()).collect()
}

impl Display for UnweightedConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnweightedConstraint::AllAdjacencies { the_matches, .. } => {
                write!(
                    f,
                    "All({})",
                    the_matches
                        .iter()
                        .map(|(_, i, j)| format!("({i}, {j}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            UnweightedConstraint::Adjacency { other_ports } => {
                write!(
                    f,
                    "Adjacency({:?}, {:?})",
                    other_ports.addr.the_match, other_ports.label
                )
            }
            UnweightedConstraint::Dangling => write!(f, "Dangling"),
        }
    }
}

fn adjacency_constraint(
    g: &PortGraph,
    this_ports: Vec<PortIndex>,
    other_ports: Vec<PortIndex>,
) -> bool {
    this_ports.into_iter().any(|p| {
        let Some(other_p) = g.port_link(p) else { return false };
        other_ports.contains(&other_p)
    })
}

impl<'g> Skeleton<'g> {
    pub(crate) fn get_port_addr(&self, port: PortIndex) -> Address {
        let node = self.graph().port_node(port).expect("invalid pattern");
        let offset = self.graph().port_offset(port).expect("invalid pattern");
        Address {
            addr: self.get_node_addr(node),
            label: match self.graph().port_direction(port).expect("invalid pattern") {
                Direction::Incoming => PortLabel::Incoming(offset.index()),
                Direction::Outgoing => PortLabel::Outgoing(offset.index()),
            },
        }
    }

    pub(crate) fn get_node_addr(&self, node: NodeIndex) -> NodeAddress {
        if node == self.root {
            return NodeAddress {
                no_match: Vec::new(),
                the_match: (Vec::new(), 0, 0),
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
        let mut ribs = self.get_ribs(&self.spine);
        let mut spine = self.spine.clone();
        let the_match = (spine[addr.0].0.clone(), spine[addr.0].1, addr.1);
        match addr.1.cmp(&0) {
            cmp::Ordering::Greater => {
                spine.truncate(addr.0 + 1);
                ribs.truncate(addr.0 + 1);
                ribs[addr.0] = [0, addr.1 - 1];
            }
            cmp::Ordering::Less => {
                spine.truncate(addr.0 + 1);
                ribs.truncate(addr.0 + 1);
                ribs[addr.0][0] = addr.1 + 1;
            }
            cmp::Ordering::Equal => {
                spine.truncate(addr.0);
                ribs.truncate(addr.0);
            }
        }
        let no_match = spine
            .into_iter()
            .zip(ribs)
            .map(|((fst, snd), third)| (fst, snd, third))
            .collect();
        NodeAddress {
            the_match,
            no_match,
        }
    }
}
