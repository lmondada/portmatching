use std::{
    collections::{BTreeSet, VecDeque},
    hash::Hash,
};

use super::{Line, LinePattern};
use crate::{EdgeProperty, HashMap, HashSet, NodeProperty, Universe};
use itertools::Itertools;
use petgraph::visit::{GraphBase, IntoNodeIdentifiers};
use portgraph::{Direction, LinkView, NodeIndex, PortOffset, SecondaryMap};

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pattern<U: Universe, PNode, PEdge: Eq + Hash> {
    nodes: HashMap<U, PNode>,
    edges: HashMap<(U, PEdge), U>,
    root: Option<U>,
}

pub type UnweightedPattern = Pattern<NodeIndex, (), UnweightedEdge>;
pub type WeightedPattern<W> = Pattern<NodeIndex, W, UnweightedEdge>;

pub(crate) type UnweightedEdge = (PortOffset, PortOffset);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub(crate) struct Edge<U, PNode, PEdge> {
    pub(crate) source: Option<U>,
    pub(crate) target: Option<U>,
    pub(crate) edge_prop: PEdge,
    pub(crate) source_prop: Option<PNode>,
    pub(crate) target_prop: Option<PNode>,
}

impl<'a, U: Universe, PNode: Clone, PEdge: Clone> Edge<U, &'a PNode, &'a PEdge> {
    pub(crate) fn to_owned_props(&self) -> Edge<U, PNode, PEdge> {
        Edge {
            source: self.source,
            target: self.target,
            edge_prop: self.edge_prop.clone(),
            source_prop: self.source_prop.cloned(),
            target_prop: self.target_prop.cloned(),
        }
    }
}

#[derive(Debug)]
struct NoRootFound;

impl<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty> Pattern<U, PNode, PEdge> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn require(&mut self, node: U, property: PNode) {
        self.nodes.insert(node, property);
    }

    pub fn add_edge(&mut self, u: U, v: U, property: PEdge) {
        self.edges.insert((u, property), v);
    }

    pub fn root(&self) -> Option<U> {
        self.root
    }

    pub fn set_root(&mut self, root: U) {
        self.root = Some(root);
    }

    pub fn dot_string(&self) -> String
    where
        U: std::fmt::Debug,
        PNode: std::fmt::Debug,
        PEdge: std::fmt::Debug,
    {
        let mut s = String::new();
        s.push_str("digraph {\n");
        for (u, property) in &self.nodes {
            s.push_str(&format!("  {:?} [label=\"{:?}\"];\n", u, property));
        }
        for ((u, property), v) in &self.edges {
            s.push_str(&format!(
                "  {:?} -> {:?} [label=\"{:?}\"];\n",
                u, v, property
            ));
        }
        s.push_str("}\n");
        s
    }

    /// Let the pattern fix a root
    ///
    /// We require `Ord` so that this is deterministic (do not rely on hash)
    fn set_any_root(&mut self) -> Result<U, NoRootFound>
    where
        U: Ord,
    {
        let all_edges: BTreeSet<_> = self
            .edges
            .iter()
            .map(|((u, property), &v)| Edge {
                source: Some(*u),
                target: Some(v),
                edge_prop: property,
                source_prop: self.nodes.get(&u),
                target_prop: self.nodes.get(&v),
            })
            .collect();
        let root = self
            .all_nodes()
            .find(|&root| order_edges(&all_edges, root).is_some())
            .ok_or(NoRootFound)?;
        self.root = Some(root);
        Ok(root)
    }

    fn all_nodes(&self) -> impl Iterator<Item = U> + '_ {
        self.nodes
            .keys()
            .copied()
            .chain(self.edges.iter().flat_map(|(&(u, _), &v)| vec![u, v]))
            .unique()
    }

    /// The edges of the pattern, in a connected order (if it exists)
    ///
    /// If no root was set, this returns `None`
    pub(crate) fn edges<'a>(&'a self) -> Option<Vec<Edge<U, &'a PNode, &'a PEdge>>> {
        let all_edges: BTreeSet<_> = self
            .edges
            .iter()
            .map(|((u, property), &v)| Edge {
                source: Some(*u),
                target: Some(v),
                edge_prop: property,
                source_prop: self.nodes.get(&u),
                target_prop: self.nodes.get(&v),
            })
            .collect();

        order_edges(&all_edges, self.root?)
    }
}

fn order_edges<'a, U: Universe, PNode: NodeProperty, PEdge: EdgeProperty>(
    all_edges: &BTreeSet<Edge<U, &'a PNode, &'a PEdge>>,
    root_candidate: U,
) -> Option<Vec<Edge<U, &'a PNode, &'a PEdge>>> {
    let mut known_nodes = HashSet::default();
    let mut known_edges = HashSet::default();
    known_nodes.insert(root_candidate);

    let mut edges = Vec::new();
    while edges.len() < all_edges.len() {
        let next_edge = all_edges
            .iter()
            .filter(|e| !known_edges.contains(e))
            .find(|e| {
                let src = e.source.expect("Pattern cannot have dangling edges");
                let tgt = e.target.expect("Pattern cannot have dangling edges");
                let rev = e.edge_prop.reverse().is_some();
                known_nodes.contains(&src) || (rev && known_nodes.contains(&tgt))
            })?;
        edges.push(*next_edge);
        known_nodes.insert(next_edge.source.unwrap());
        known_nodes.insert(next_edge.target.unwrap());
        known_edges.insert(next_edge);
    }
    Some(edges)
}

impl<U: Universe, PNode, PEdge: Eq + Hash> Default for Pattern<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
            root: None,
        }
    }
}

fn add_nodes<G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>>(
    pattern: &mut Pattern<NodeIndex, (), (PortOffset, PortOffset)>,
    g: G,
) {
    for n in g.node_identifiers() {
        pattern.require(n, ());
    }
}

fn add_nodes_weighted<G, W>(
    pattern: &mut Pattern<NodeIndex, W, (PortOffset, PortOffset)>,
    g: G,
    w: impl SecondaryMap<NodeIndex, W>,
) where
    G: IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
    W: NodeProperty,
{
    for n in g.node_identifiers() {
        pattern.require(n, w.get(n).clone());
    }
}

fn add_edges<W, G>(pattern: &mut Pattern<NodeIndex, W, (PortOffset, PortOffset)>, g: G)
where
    W: NodeProperty,
    G: LinkView,
{
    for p in g.ports_iter() {
        if g.port_offset(p).unwrap().direction() == Direction::Incoming {
            continue;
        }
        let pout = p;
        let Some(pin) = g.port_link(pout) else {
            continue
        };
        let pout_offset = g.port_offset(pout).unwrap();
        let pin_offset = g.port_offset(pin).unwrap();
        let pout_node = g.port_node(pout).unwrap();
        let pin_node = g.port_node(pin).unwrap();
        pattern.add_edge(pout_node, pin_node, (pout_offset, pin_offset));
    }
}

impl Pattern<NodeIndex, (), (PortOffset, PortOffset)> {
    pub fn from_portgraph<G>(g: G) -> Self
    where
        G: LinkView + IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
    {
        let mut pattern = Self::new();
        add_nodes(&mut pattern, g);
        add_edges(&mut pattern, g);
        pattern.set_any_root().expect("Could not find root");
        pattern
    }

    pub fn from_rooted_portgraph<G>(g: G, root: NodeIndex) -> Self
    where
        G: LinkView + IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
    {
        let mut pattern = Self::new();
        add_nodes(&mut pattern, g);
        add_edges(&mut pattern, g);
        pattern.set_root(root);
        pattern
    }
}

impl<W: NodeProperty> Pattern<NodeIndex, W, (PortOffset, PortOffset)> {
    pub fn from_weighted_portgraph<G>(g: G, w: impl SecondaryMap<NodeIndex, W>) -> Self
    where
        G: LinkView + IntoNodeIdentifiers + GraphBase<NodeId = NodeIndex>,
    {
        let mut pattern = Self::new();
        add_nodes_weighted(&mut pattern, g, w);
        add_edges(&mut pattern, g);
        pattern.set_any_root().expect("Could not find root");
        pattern
    }
}

impl<U: Universe, PNode, PEdge: EdgeProperty> Pattern<U, PNode, PEdge> {
    /// Try to convert the pattern into a line pattern
    ///
    /// Attempt every possible root and return `None` if none worked.
    pub(crate) fn try_into_line_pattern<F: for<'a> Fn(&'a PEdge, &'a PEdge) -> bool>(
        self,
        valid_successor: F,
    ) -> Option<LinePattern<U, PNode, PEdge>> {
        let Self {
            nodes,
            mut edges,
            root,
        } = self;
        let mut to_visit = VecDeque::new();

        // Add all the valid reverse edges too
        let rev_edges = edges
            .iter()
            .filter_map(|((u, p), &v)| p.reverse().map(|rev| ((v, rev), *u)))
            .collect_vec();
        edges.extend(rev_edges);

        add_new_edges(&mut to_visit, root?, edges.keys());
        let mut lines = Vec::new();
        while let Some((u, property)) = to_visit.pop_front() {
            let mut new_edges = Vec::new();
            let mut curr_edge = (u, property);
            loop {
                let (u, property) = curr_edge;
                let Some(v) = edges.remove(&(u, property.clone())) else {
                    break;
                };
                // Also remove the reverse edge (if present)
                if let Some(rev) = property.reverse() {
                    edges.remove(&(v, rev));
                }
                new_edges.push((u, v, property.clone()));
                add_new_edges(&mut to_visit, v, edges.keys());
                let Some((_, new_prop)) = edges.keys()
                    .find(|(u, p)| u == &v && valid_successor(&property, p))
                    .cloned()
                else { break };
                curr_edge = (v, new_prop);
            }
            if !new_edges.is_empty() {
                let line = Line::new(u, new_edges);
                lines.push(line);
            }
        }
        edges.is_empty().then_some(LinePattern { nodes, lines })
    }
}

fn add_new_edges<'a, U: Universe + 'a, PEdge: EdgeProperty + 'a>(
    queue: &mut VecDeque<(U, PEdge)>,
    node: U,
    edges: impl IntoIterator<Item = &'a (U, PEdge)>,
) {
    let mut node_edges = edges
        .into_iter()
        .cloned()
        .filter(|&(u, _)| u == node)
        .collect_vec();
    node_edges.sort_unstable_by_key(|(_, prop)| prop.clone());
    queue.extend(node_edges)
}

pub(crate) fn compatible_offsets(
    (_, pout): &(PortOffset, PortOffset),
    (pin, _): &(PortOffset, PortOffset),
) -> bool {
    pout.direction() != pin.direction() && pout.index() == pin.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_line_pattern() {
        let mut p: Pattern<_, (), _> = Default::default();
        let po = PortOffset::new_outgoing;
        let pi = PortOffset::new_incoming;
        p.add_edge(0, 1, (po(0), pi(1)));
        p.add_edge(1, 2, (po(1), pi(0)));
        p.add_edge(0, 1, (po(1), pi(0)));
        p.add_edge(1, 2, (po(0), pi(1)));
        p.add_edge(0, 1, (po(2), pi(2)));
        p.add_edge(1, 2, (po(2), pi(2)));
        p.set_root(0);
        assert_eq!(
            p.try_into_line_pattern(compatible_offsets),
            Some(LinePattern {
                nodes: HashMap::default(),
                lines: vec![
                    Line::new(0, vec![(0, 1, (po(0), pi(1))), (1, 2, (po(1), pi(0))),]),
                    Line::new(0, vec![(0, 1, (po(1), pi(0))), (1, 2, (po(0), pi(1))),]),
                    Line::new(0, vec![(0, 1, (po(2), pi(2))), (1, 2, (po(2), pi(2))),])
                ]
            })
        );
    }

    #[test]
    fn from_pattern_with_rev() {
        let mut p = Pattern::<_, (), _>::new();
        let po = PortOffset::new_outgoing;
        let pi = PortOffset::new_incoming;
        p.add_edge(0, 0, (po(0), pi(2)));
        p.add_edge(0, 1, (po(1), pi(1)));
        p.add_edge(2, 0, (po(0), pi(1)));
        p.set_root(0);
        let lp = p
            .try_into_line_pattern(compatible_offsets)
            .expect("Could not convert to line pattern");
        assert_eq!(
            lp.lines,
            vec![
                Line::new(0, vec![(0, 2, (pi(1), po(0)))]),
                Line::new(0, vec![(0, 0, (pi(2), po(0)))]),
                Line::new(0, vec![(0, 1, (po(1), pi(1)))]),
            ]
        );
    }

    #[test]
    fn from_pattern_simple() {
        let mut p = Pattern::<_, (), _>::new();
        p.add_edge(0, 0, (0, 2));
        p.add_edge(0, 1, (1, 1));
        p.add_edge(0, 2, (2, 0));
        p.set_root(2);
        p.try_into_line_pattern(|(_, pout), (pin, _)| pin == pout)
            .expect("Could not convert to line pattern");
    }

    #[test]
    fn from_pattern2() {
        let mut p = Pattern::<_, (), _>::new();
        let po = PortOffset::new_outgoing;
        let pi = PortOffset::new_incoming;
        let mut add_edge = |(i, j), (k, l)| p.add_edge(i, k, (po(j), pi(l)));
        add_edge((0, 0), (1, 0));
        add_edge((2, 0), (1, 1));
        p.set_any_root().expect("Could not pick any root");
        p.try_into_line_pattern(|(_, pout), (pin, _)| pin == pout)
            .expect("Could not convert to line pattern");
    }
}
