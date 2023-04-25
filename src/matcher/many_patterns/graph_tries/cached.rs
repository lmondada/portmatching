use std::{
    collections::{BTreeMap, HashMap},
    fmt::{self, Display},
};

use portgraph::{Direction, NodeIndex, PortGraph, PortIndex, PortOffset, Weights};

use crate::addresses::{follow_path, port_opposite, LinePartition, LinePoint, Ribs};

use super::{
    BaseGraphTrie, BoundedAddress, CacheOption, GraphCache, GraphTrie, StateID, StateTransition,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SpineID(usize);

struct SpineCache {
    cache: Vec<CacheOption<SpinePoint>>,
}

#[derive(Clone)]
struct SpinePoint {
    line_ind: usize,
    ind: isize,
    offset: usize,
    node: NodeIndex,
    spine_id: SpineID,
}

impl SpinePoint {
    fn new(lp: &LinePoint, graph: &PortGraph, spine_id: SpineID) -> Self {
        let port = lp.in_port.or(lp.out_port).unwrap_or(PortIndex::new(0));
        let offset = graph.port_offset(port).expect("invalid port").index();
        let node = graph.port_node(port).expect("invalid port");
        SpinePoint {
            line_ind: lp.line_ind,
            ind: lp.ind,
            offset,
            node,
            spine_id,
        }
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
                line_ind: 0,
                ind: 0,
                offset: 0,
                node: root,
                spine_id: SpineID(0),
            })
            .into()],
        }
    }

    fn get(&self, spine: SpineID) -> Option<&SpinePoint> {
        self.cache.get(spine.0)?.as_ref().to_option()
    }

    fn get_or_insert_with<'a, F: FnOnce() -> (&'a Vec<PortOffset>, usize)>(
        &'a mut self,
        spine: SpineID,
        find_spine: F,
        node2line: &BTreeMap<NodeIndex, Vec<LinePoint>>,
        graph: &PortGraph,
        root: NodeIndex,
    ) -> Option<&SpinePoint> {
        if self.cache.len() <= spine.0 {
            self.cache.resize(spine.0 + 1, CacheOption::NoCache);
        }
        if self.cache[spine.0].no_cache() {
            let (path, offset) = find_spine();
            let n = follow_path(path, root, graph)?;
            let lp = node2line[&n].iter().find(|line| {
                for port in [line.out_port, line.in_port].into_iter().flatten() {
                    let out_port = graph.port_offset(port).expect("invalid port");
                    if out_port.index() == offset {
                        return true;
                    }
                }
                false
            })?;
            self.cache[spine.0] = Some(SpinePoint::new(lp, graph, spine)).into();
        }
        self.cache[spine.0].as_ref().cached()
    }
}

pub struct AddressCache<'graph> {
    spine: SpineCache,
    address_cache: HashMap<Address, NodeIndex>,
    node2line: BTreeMap<NodeIndex, Vec<LinePoint>>,
    graph: &'graph PortGraph,
    root: NodeIndex,
}

impl<'a> AddressCache<'a> {
    fn get_all_addresses(
        &mut self,
        node: NodeIndex,
        spine: &Vec<(SpineID, Vec<PortOffset>, usize)>,
    ) -> Vec<(SpineID, usize, isize)> {
        if node == self.root {
            return vec![(SpineID(0), 0, 0)];
        }
        self.insert_spine(spine);
        let spine = self.get_spine(spine.iter().map(|(spine_id, _, _)| *spine_id));
        let mut rev_inds: BTreeMap<_, Vec<_>> = Default::default();
        for (i, lp) in spine.iter().enumerate() {
            if let Some(lp) = lp {
                rev_inds.entry(lp.line_ind).or_default().push(i);
            }
        }
        let mut all_addrs = Vec::new();
        for line in self.node2line[&node].iter() {
            for &spine_ind in rev_inds.get(&line.line_ind).unwrap_or(&Vec::new()) {
                let spine = spine[spine_ind]
                    .as_ref()
                    .expect("By construction of in rev_inds");
                let ind = line.ind - spine.ind;
                all_addrs.push((spine.spine_id, spine_ind, ind))
            }
        }
        all_addrs.sort_unstable();
        all_addrs
    }

    fn insert_spine(&mut self, spine: &Spine) {
        for (spine_id, path, offset) in spine.iter() {
            self.spine.get_or_insert_with(
                *spine_id,
                || (path, *offset),
                &self.node2line,
                self.graph,
                self.root,
            );
        }
    }

    fn get_spine<I: Iterator<Item = SpineID>>(&self, spine_ids: I) -> Vec<Option<&SpinePoint>> {
        spine_ids.map(|spine_id| self.spine.get(spine_id)).collect()
    }
}

impl<'graph> GraphCache<'graph, AddressWithBound> for AddressCache<'graph> {
    fn init(graph: &'graph PortGraph, root: NodeIndex) -> Self {
        let lp = LinePartition::new(graph, root);
        Self {
            spine: SpineCache::new(root),
            address_cache: HashMap::new(),
            node2line: lp
                .node2line
                .into_iter()
                .enumerate()
                .map(|(i, v)| (NodeIndex::new(i), v))
                .collect(),
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
                    .get_or_insert_with(
                        *spine_id,
                        find_spine,
                        &self.node2line,
                        self.graph,
                        self.root,
                    )
                    .expect("could not find spine");
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
        let all_addrs = self.get_all_addresses(node, boundary.spine.as_ref()?);
        let addrs = if let Some(Ribs(ribs)) = boundary.ribs.as_ref() {
            all_addrs.into_iter().find(|&(_, spine_ind, ind)| {
                let Some(&[from, to]) = ribs.get(spine_ind) else {
                return false
            };
                from <= ind && to >= ind
            })
        } else {
            all_addrs.into_iter().next()
        };
        addrs.map(|(spine_id, _, ind)| (spine_id, ind))
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

#[derive(Clone, PartialEq, Eq)]
pub struct AddressWithBound {
    address: Address,
    boundary: Skeleton,
}

impl<'a> BoundedAddress<'a> for AddressWithBound {
    type Main = Address;
    type Boundary = Skeleton;
    type Cache = AddressCache<'a>;

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

pub struct CachedGraphTrie {
    pub(crate) graph: PortGraph,
    weights: Weights<NodeWeight, StateTransition<(Address, Ribs)>>,
}

impl CachedGraphTrie {
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

impl<'graph> GraphTrie<'graph> for CachedGraphTrie {
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
