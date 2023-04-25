use std::{
    collections::HashMap,
    fmt::{self, Display},
    iter::repeat,
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset, Weights};

use crate::addresses::{follow_path, port_opposite, Ribs};

use super::{
    BaseGraphTrie, BoundedAddress, CacheOption, GraphCache, GraphTrie, StateID, StateTransition,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SpineID(usize);

struct SpineCache {
    cache: Vec<CacheOption<SpinePoint>>,
}

#[derive(Clone, Debug)]
struct SpinePoint {
    node: NodeIndex,
    offset: usize,
}

impl SpinePoint {
    fn new(node: NodeIndex, offset: usize) -> Self {
        SpinePoint { offset, node }
    }

    fn out_port(&self, graph: &PortGraph) -> Option<PortIndex> {
        graph.output(self.node, self.offset)
    }

    fn in_port(&self, graph: &PortGraph) -> Option<PortIndex> {
        graph.input(self.node, self.offset)
    }
}

impl SpineCache {
    fn new(root: NodeIndex) -> Self {
        Self {
            cache: vec![Some(SpinePoint {
                offset: 0,
                node: root,
            })
            .into()],
        }
    }

    fn get_or_insert_with<'a, F: FnOnce() -> (&'a Vec<PortOffset>, usize)>(
        &'a mut self,
        spine: SpineID,
        find_spine: F,
        graph: &PortGraph,
        root: NodeIndex,
    ) -> Option<&SpinePoint> {
        if self.cache.len() <= spine.0 {
            self.cache.resize(spine.0 + 1, CacheOption::NoCache);
        }
        if self.cache[spine.0].no_cache() {
            let (path, offset) = find_spine();
            let n = follow_path(path, root, graph)?;
            self.cache[spine.0] = Some(SpinePoint::new(n, offset)).into();
        }
        self.cache[spine.0].as_ref().cached()
    }
}

pub struct AddressNoCache<'graph> {
    spine: SpineCache,
    address_cache: HashMap<Address, NodeIndex>,
    graph: &'graph PortGraph,
    root: NodeIndex,
}

impl<'graph> GraphCache<'graph, AddressWithBound> for AddressNoCache<'graph> {
    fn init(graph: &'graph PortGraph, root: NodeIndex) -> Self {
        Self {
            spine: SpineCache::new(root),
            address_cache: HashMap::new(),
            graph,
            root,
        }
    }

    fn get_node(
        &mut self,
        (spine_id, ind): &<AddressWithBound as BoundedAddress>::Main,
        boundary: &<AddressWithBound as BoundedAddress>::Boundary,
    ) -> Option<NodeIndex> {
        self.address_cache
            .get(&(*spine_id, *ind))
            .copied()
            .or_else(|| {
                let find_spine = || {
                    let lp = boundary
                        .spine
                        .as_ref()
                        .expect("invalid spine")
                        .iter()
                        .find(|(s, _, _)| s == spine_id)
                        .expect("invalid spine ID");
                    (&lp.1, lp.2)
                };
                let sp = self
                    .spine
                    .get_or_insert_with(*spine_id, find_spine, self.graph, self.root)?;
                let mut port = match *ind {
                    ind if ind > 0 => sp.out_port(self.graph),
                    ind if ind < 0 => sp.in_port(self.graph),
                    _ => sp.out_port(self.graph).or(sp.in_port(self.graph)),
                };
                let mut node = self.graph.port_node(port?).expect("invalid port");
                for _ in 0..ind.abs() {
                    let next_port = self.graph.port_link(port?)?;
                    node = self.graph.port_node(next_port).expect("invalid port");
                    port = port_opposite(next_port, self.graph);
                }
                self.address_cache.insert((*spine_id, *ind), node);
                Some(node)
            })
    }

    fn get_addr(
        &mut self,
        node: NodeIndex,
        boundary: &<AddressWithBound as BoundedAddress>::Boundary,
    ) -> Option<<AddressWithBound as BoundedAddress>::Main> {
        // let all_addrs = self.get_all_addresses(node, boundary.spine.as_ref()?);
        if node == self.root {
            return Some((SpineID(0), 0));
        }
        let spine = boundary.spine.as_ref()?;
        let ribs = boundary.ribs_iter();
        for (&(spine_id, ref path, offset), rib) in spine.iter().zip(ribs) {
            let sp =
                self.spine
                    .get_or_insert_with(spine_id, || (path, offset), self.graph, self.root);
            if let Some(sp) = sp {
                let &[bef, aft] = rib.unwrap_or(&[isize::MIN, isize::MAX]);
                let mut ind = 0;
                let mut port = sp.out_port(self.graph);
                if sp.node == node && ind >= bef && ind <= aft {
                    return Some((spine_id, ind));
                }
                while port.is_some() && ind < aft {
                    port = self.graph.port_link(port.unwrap());
                    ind += 1;
                    if let Some(port_some) = port {
                        let curr_node = self.graph.port_node(port_some).expect("invalid port");
                        if curr_node == node && ind >= bef {
                            return Some((spine_id, ind));
                        }
                        port = port_opposite(port_some, self.graph);
                    }
                }
                port = sp.in_port(self.graph);
                ind = 0;
                while port.is_some() && ind > bef {
                    port = self.graph.port_link(port.unwrap());
                    ind -= 1;
                    if let Some(port_some) = port {
                        let curr_node = self.graph.port_node(port_some).expect("invalid port");
                        if curr_node == node && ind <= aft {
                            return Some((spine_id, ind));
                        }
                        port = port_opposite(port_some, self.graph);
                    }
                }
            }
        }
        None
    }

    fn graph(&self) -> &'graph PortGraph {
        self.graph
    }
}

type Address = (SpineID, isize);
type Spine = Vec<(SpineID, Vec<PortOffset>, usize)>;

#[derive(Clone, PartialEq, Eq)]
pub struct Skeleton {
    spine: Option<Spine>,
    ribs: Option<Ribs>,
}

impl Skeleton {
    fn ribs_iter(&self) -> impl Iterator<Item = Option<&[isize; 2]>> + '_ {
        self.ribs
            .as_ref()
            .map(|r| {
                Box::new(r.0.iter().map(Some)) as Box<dyn Iterator<Item = Option<&[isize; 2]>>>
            })
            .unwrap_or_else(|| Box::new(repeat(None)))
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct AddressWithBound {
    address: Address,
    boundary: Skeleton,
}

impl<'a> BoundedAddress<'a> for AddressWithBound {
    type Main = Address;
    type Boundary = Skeleton;
    type Cache = AddressNoCache<'a>;

    fn boundary(&self) -> &Self::Boundary {
        &self.boundary
    }

    fn main(&self) -> &Self::Main {
        &self.address
    }
}

#[derive(Clone, Default, Debug)]
struct NodeWeight {
    out_port: Option<PortOffset>,
    address: Option<Address>,
    spine: Option<Spine>,
    non_deterministic: bool,
}

impl Display for NodeWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(addr) = &self.address {
            write!(f, "{:?}", (addr.0 .0, addr.1))?;
        } else {
            write!(f, "None")?;
        }
        if let Some(port) = &self.out_port {
            write!(f, "[{port:?}]")?;
        }
        Ok(())
    }
}

pub struct NoCachedGraphTrie {
    pub(crate) graph: PortGraph,
    weights: Weights<NodeWeight, StateTransition<(Address, Ribs)>>,
}

impl NoCachedGraphTrie {
    pub fn new(base: &BaseGraphTrie) -> Self {
        let mut weights = Weights::new();
        let mut existing_spines = HashMap::new();
        let mut next_ind = 0;
        for node in base.graph.nodes_iter() {
            let weight: &mut NodeWeight = weights.nodes.get_mut(node);
            weight.out_port = base.weight(node).out_port;
            if let Some(spine) = base.weight(node).spine.as_ref() {
                let mut new_spine = Vec::with_capacity(spine.len());
                for s in spine {
                    let &mut spine_id =
                        existing_spines
                            .entry((s.0.clone(), s.1))
                            .or_insert_with(|| {
                                let ret = next_ind;
                                next_ind += 1;
                                SpineID(ret)
                            });
                    new_spine.push((spine_id, s.0.clone(), s.1));
                }
                weights[node].spine = Some(new_spine);
            }
            if let Some(addr) = base.weights[node].address.as_ref() {
                let spine_id = weights[node]
                    .spine
                    .as_ref()
                    .expect("cannot compute address")[addr.0]
                    .0;
                let new_addr = (spine_id, addr.1);
                weights[node].address = Some(new_addr);
            }
            weights[node].non_deterministic = base.weights[node].non_deterministic;
            for port in base.graph.outputs(node) {
                let weight: &mut StateTransition<(Address, Ribs)> = weights.ports.get_mut(port);
                match &base.weights[port] {
                    StateTransition::Node(addrs, out_port) => {
                        let mut new_addrs = Vec::with_capacity(addrs.len());
                        for (addr, ribs) in addrs {
                            let spine_id = weights.nodes[node]
                                .spine
                                .as_ref()
                                .expect("cannot compute address")[addr.0]
                                .0;
                            let new_addr = (spine_id, addr.1);
                            new_addrs.push((new_addr, ribs.clone()));
                        }
                        *weight = StateTransition::Node(new_addrs, *out_port);
                    }
                    StateTransition::NoLinkedNode => {
                        *weight = StateTransition::NoLinkedNode;
                    }
                    StateTransition::FAIL => {
                        *weight = StateTransition::FAIL;
                    }
                }
            }
        }
        Self {
            graph: base.graph.clone(),
            weights,
        }
    }

    pub(crate) fn str_weights(&self) -> Weights<String, String> {
        let mut str_weights = Weights::new();
        for p in self.graph.ports_iter() {
            str_weights[p] = match self.graph.port_direction(p).unwrap() {
                Direction::Incoming => "".to_string(),
                Direction::Outgoing => self.weights[p].to_string(),
            }
        }
        for n in self.graph.nodes_iter() {
            str_weights[n] = self.weights[n].to_string();
        }
        str_weights
    }
}

impl<'graph> GraphTrie<'graph> for NoCachedGraphTrie {
    type Address = AddressWithBound;

    fn trie(&self) -> &portgraph::PortGraph {
        &self.graph
    }

    fn address(&self, state: StateID) -> Option<Self::Address> {
        self.weights[state].address.map(|address| AddressWithBound {
            address,
            boundary: Skeleton {
                spine: self.weights[state].spine.clone(),
                ribs: None,
            },
        })
    }

    fn port_offset(&self, state: StateID) -> Option<portgraph::PortOffset> {
        self.weights[state].out_port
    }

    fn transition(&self, port: portgraph::PortIndex) -> StateTransition<Self::Address> {
        let node = self.graph.port_node(port).expect("invalid port");
        let spine = self.weights[node].spine.as_ref();
        match &self.weights[port] {
            StateTransition::Node(addrs, port) => StateTransition::Node(
                addrs
                    .iter()
                    .map(|(addr, ribs)| AddressWithBound {
                        address: *addr,
                        boundary: Skeleton {
                            spine: spine.cloned(),
                            ribs: Some(ribs.clone()),
                        },
                    })
                    .collect(),
                *port,
            ),
            StateTransition::NoLinkedNode => StateTransition::NoLinkedNode,
            StateTransition::FAIL => StateTransition::FAIL,
        }
    }

    fn is_non_deterministic(&self, state: StateID) -> bool {
        self.weights[state].non_deterministic
    }
}
