use std::{collections::BTreeSet, fmt::Display, mem};

use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset};

use crate::utils::{follow_path, port_opposite};

use super::AddressCache;

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
    pub fn node<C: AddressCache>(
        &self,
        g: &PortGraph,
        root: NodeIndex,
        _cache: &mut C,
    ) -> Option<NodeIndex> {
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
pub struct PortAddress {
    pub(super) addr: NodeAddress,
    pub(super) label: PortLabel,
}

impl PortAddress {
    pub(crate) fn ports(&self, g: &PortGraph, root: NodeIndex) -> Vec<PortIndex> {
        let Some(node) = self.addr.node(g, root, &mut ()) else { return vec![] };
        let as_vec = |p: Option<_>| p.into_iter().collect();
        match self.label {
            PortLabel::Outgoing(out_p) => as_vec(g.output(node, out_p)),
            PortLabel::Incoming(in_p) => as_vec(g.input(node, in_p)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Constraint {
    // All constraints must be satisfied
    AllAdjacencies {
        label: PortLabel,
        no_match: Vec<(Vec<PortOffset>, usize, [isize; 2])>,
        the_matches: Vec<(Vec<PortOffset>, usize, isize)>,
    },
    // Port must be linked to one of `other_ports`
    Adjacency {
        other_ports: PortAddress,
    },
    // Port must be dangling (at least existing)
    Dangling,
}

impl Constraint {
    pub fn is_satisfied(&self, this_ports: &PortAddress, g: &PortGraph, root: NodeIndex) -> bool {
        match self {
            Constraint::Adjacency { other_ports } => {
                let other_ports = other_ports.ports(g, root);
                let this_ports = this_ports.ports(g, root);
                adjacency_constraint(g, this_ports, other_ports)
            }
            Constraint::Dangling => !this_ports.ports(g, root).is_empty(),
            Constraint::AllAdjacencies { .. } => self
                .to_vec()
                .iter()
                .all(|c| c.is_satisfied(this_ports, g, root)),
        }
    }

    fn to_vec(&self) -> Vec<Self> {
        let Constraint::AllAdjacencies {
            label,
            no_match,
            the_matches,
        } = self else { return vec![self.clone()] };
        let mut no_match = no_match.clone();
        the_matches
            .iter()
            .map(|m| Constraint::Adjacency {
                other_ports: PortAddress {
                    addr: NodeAddress {
                        no_match: mem::take(&mut no_match),
                        the_match: m.clone(),
                    },
                    label: *label,
                },
            })
            .collect()
    }

    // Merge two constraints
    pub fn and(&self, other: &Constraint) -> Option<Constraint> {
        // Gather all constraints in a vec
        simplify_constraints(vec![self.clone(), other.clone()])
    }
}

fn simplify_constraints(constraints: Vec<Constraint>) -> Option<Constraint> {
    let constraints = flatten_constraints(constraints);

    // If we only have dangling, then dangling. Otherwise we can remove danglings
    if constraints.iter().all(|c| c == &Constraint::Dangling) {
        return Some(Constraint::Dangling);
    }

    let mut matches = BTreeSet::new();
    let mut no_matches = BTreeSet::new();
    let mut all_labels = Vec::new();
    for c in constraints {
        if let Constraint::Adjacency { other_ports: ports } = c {
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
        1 => Some(Constraint::Adjacency {
            other_ports: PortAddress {
                addr: NodeAddress {
                    no_match: no_matches.into_iter().collect(),
                    the_match: matches.into_iter().next().unwrap(),
                },
                label,
            },
        }),
        _ => Some(Constraint::AllAdjacencies {
            label,
            no_match: no_matches.into_iter().collect(),
            the_matches: matches.into_iter().collect(),
        }),
    }
}

fn flatten_constraints(constraints: Vec<Constraint>) -> Vec<Constraint> {
    constraints.into_iter().flat_map(|c| c.to_vec()).collect()
}

impl Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constraint::AllAdjacencies { the_matches, .. } => {
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
            Constraint::Adjacency { other_ports } => {
                write!(
                    f,
                    "Adjacency({:?}, {:?})",
                    other_ports.addr.the_match, other_ports.label
                )
            }
            Constraint::Dangling => write!(f, "Dangling"),
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
