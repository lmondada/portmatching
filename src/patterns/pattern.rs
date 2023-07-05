use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
};

use super::{Line, LinePattern};
use crate::{EdgeProperty, NodeProperty, Universe};
use itertools::Itertools;
use portgraph::{Direction, LinkView, NodeIndex, PortGraph, PortOffset, PortView};

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pattern<U: Universe, PNode, PEdge: Eq + Hash> {
    nodes: HashMap<U, PNode>,
    edges: HashMap<(U, PEdge), U>,
    root: Option<U>,
}

pub type UnweightedPattern = Pattern<NodeIndex, (), UnweightedEdge>;

pub(crate) type UnweightedEdge = (PortOffset, PortOffset);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct Edge<U, PNode, PEdge> {
    pub(crate) source: Option<U>,
    pub(crate) target: Option<U>,
    pub(crate) edge_prop: PEdge,
    pub(crate) source_prop: Option<PNode>,
    pub(crate) target_prop: Option<PNode>,
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

    /// Let the pattern fix a root
    fn set_any_root(&mut self) -> Result<U, NoRootFound> {
        let all_edges: HashSet<_> = self
            .edges
            .iter()
            .map(|(&(u, property), &v)| Edge {
                source: Some(u),
                target: Some(v),
                edge_prop: property,
                source_prop: self.nodes.get(&u).copied(),
                target_prop: self.nodes.get(&v).copied(),
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
    pub(crate) fn edges(&self) -> Option<Vec<Edge<U, PNode, PEdge>>> {
        let all_edges: HashSet<_> = self
            .edges
            .iter()
            .map(|(&(u, property), &v)| Edge {
                source: Some(u),
                target: Some(v),
                edge_prop: property,
                source_prop: self.nodes.get(&u).copied(),
                target_prop: self.nodes.get(&v).copied(),
            })
            .collect();

        order_edges(&all_edges, self.root?)
    }
}

fn order_edges<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty>(
    all_edges: &HashSet<Edge<U, PNode, PEdge>>,
    root_candidate: U,
) -> Option<Vec<Edge<U, PNode, PEdge>>> {
    let mut known_nodes = HashSet::new();
    let mut known_edges = HashSet::new();
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
        edges.push(next_edge.clone());
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

impl Pattern<NodeIndex, (), (PortOffset, PortOffset)> {
    pub fn from_portgraph(g: &PortGraph) -> Self {
        let mut pattern = Self::new();
        for n in g.nodes_iter() {
            pattern.require(n, ());
        }
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
        pattern.set_any_root().expect("Could not find root");
        pattern
    }
}

impl<U: Universe, PNode, PEdge: EdgeProperty> Pattern<U, PNode, PEdge> {
    /// Try to convert the pattern into a line pattern
    ///
    /// Attempt every possible root and return `None` if none worked.
    pub(crate) fn try_into_line_pattern<F: Fn(PEdge, PEdge) -> bool>(
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
            .filter_map(|(&(u, p), &v)| p.reverse().map(|rev| ((v, rev), u)))
            .collect_vec();
        edges.extend(rev_edges);

        add_new_edges(&mut to_visit, root?, edges.keys());
        let mut lines = Vec::new();
        while let Some((u, property)) = to_visit.pop_front() {
            let mut new_edges = Vec::new();
            let mut curr_edge = (u, property);
            loop {
                let (u, property) = curr_edge;
                let Some(v) = edges.remove(&(u, property)) else {
                    break;
                };
                // Also remove the reverse edge (if present)
                if let Some(rev) = property.reverse() {
                    edges.remove(&(v, rev));
                }
                new_edges.push((u, v, property));
                add_new_edges(&mut to_visit, v, edges.keys());
                let Some(&(_, new_prop)) = edges.keys().find(|(u, p)| u == &v && valid_successor(property, *p)) else {
                    break
                };
                curr_edge = (v, new_prop);
            }
            if !new_edges.is_empty() {
                let line = Line::new(u, new_edges);
                lines.push(line);
            }
        }
        edges.is_empty().then(|| LinePattern { nodes, lines })
    }
}

fn add_new_edges<'a, U: Universe + 'a, PEdge: EdgeProperty + 'a>(
    queue: &mut VecDeque<(U, PEdge)>,
    node: U,
    edges: impl IntoIterator<Item = &'a (U, PEdge)>,
) {
    queue.extend(
        edges
            .into_iter()
            .filter(|&&(u, _)| u == node)
            .sorted_unstable_by_key(|&(_, prop)| prop),
    )
}

pub(crate) fn compatible_offsets(
    (_, pout): (PortOffset, PortOffset),
    (pin, _): (PortOffset, PortOffset),
) -> bool {
    pout.direction() != pin.direction() && pout.index() == pin.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_line_pattern() {
        let mut p: Pattern<_, (), _> = Default::default();
        let po = |i| PortOffset::new_outgoing(i);
        let pi = |i| PortOffset::new_incoming(i);
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
                nodes: HashMap::new(),
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
        let po = |i| PortOffset::new_outgoing(i);
        let pi = |i| PortOffset::new_incoming(i);
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
        let po = |i| PortOffset::new_outgoing(i);
        let pi = |i| PortOffset::new_incoming(i);
        let mut add_edge = |(i, j), (k, l)| p.add_edge(i, k, (po(j), pi(l)));
        add_edge((0, 0), (1, 0));
        add_edge((2, 0), (1, 1));
        p.set_any_root().expect("Could not pick any root");
        p.try_into_line_pattern(|(_, pout), (pin, _)| pin == pout)
            .expect("Could not convert to line pattern");
    }
}
