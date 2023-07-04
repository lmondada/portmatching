use std::collections::{HashMap, HashSet, VecDeque};

use super::{Line, LinePattern};
use crate::{Property, Universe};
use itertools::Itertools;
use portgraph::{Direction, LinkView, NodeIndex, PortGraph, PortOffset, PortView};

#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pattern<U: Universe, PNode, PEdge: Property> {
    nodes: HashMap<U, PNode>,
    edges: HashMap<(U, PEdge), U>,
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

impl<U: Universe, PNode: Property, PEdge: Property> Pattern<U, PNode, PEdge> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn require(&mut self, node: U, property: PNode) {
        self.nodes.insert(node, property);
    }

    pub fn add_edge(&mut self, u: U, v: U, property: PEdge) {
        self.edges.insert((u, property), v);
    }

    /// The edges of the pattern, in a connected order (if it exists)
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
        let all_nodes: HashSet<_> = all_edges
            .iter()
            .map(|e| vec![e.source, e.target])
            .flatten()
            .flatten()
            .collect();

        all_nodes
            .into_iter()
            .find_map(|root_candidate| order_edges(&all_edges, root_candidate))
    }
}

fn order_edges<U: Universe, PNode: Property, PEdge: Property>(
    all_edges: &HashSet<Edge<U, PNode, PEdge>>,
    root_candidate: U,
) -> Option<Vec<Edge<U, PNode, PEdge>>> {
    let mut known_nodes = HashSet::new();
    let mut known_edges = HashSet::new();
    known_nodes.insert(root_candidate);

    let mut edges = Vec::new();
    let n_edges = all_edges.len();
    while edges.len() < n_edges {
        let next_edge = all_edges
            .iter()
            .filter(|e| !known_edges.contains(e))
            .find(|e| {
                let src = e.source.expect("Pattern cannot have dangling edges");
                let tgt = e.target.expect("Pattern cannot have dangling edges");
                known_nodes.contains(&src) || known_nodes.contains(&tgt)
            })?;
        edges.push(next_edge.clone());
        known_nodes.insert(next_edge.source.unwrap());
        known_nodes.insert(next_edge.target.unwrap());
        known_edges.insert(next_edge);
    }
    Some(edges)
}

impl<U: Universe, PNode, PEdge: Property> Default for Pattern<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            nodes: Default::default(),
            edges: Default::default(),
        }
    }
}

impl Pattern<NodeIndex, (), (PortOffset, PortOffset)> {
    pub fn from_portgraph(g: &PortGraph) -> Self {
        let mut pattern = Self::new();
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
        pattern
    }
}

impl<U: Universe, PNode: Property, PEdge: Property> Pattern<U, PNode, PEdge> {
    /// Try to convert the pattern into a line pattern
    ///
    /// Attempt every possible root and return `None` if none worked.
    pub(crate) fn try_into_line_pattern<F: Fn(PEdge, PEdge) -> bool>(
        self,
        valid_successor: F,
    ) -> Option<LinePattern<U, PNode, PEdge>> {
        let mut all_nodes = self
            .edges
            .iter()
            .flat_map(|(&(u, _), &v)| vec![u, v])
            .unique();
        all_nodes.find_map(|root| {
            self.clone()
                .try_into_line_pattern_with_root(root, &valid_successor)
        })
    }

    pub(crate) fn try_into_line_pattern_with_root<F: Fn(PEdge, PEdge) -> bool>(
        self,
        root: U,
        valid_successor: F,
    ) -> Option<LinePattern<U, PNode, PEdge>> {
        let Self { nodes, mut edges } = self;
        let mut to_visit = VecDeque::new();
        add_new_edges(&mut to_visit, root, edges.keys());
        let mut lines = Vec::new();
        while let Some((u, property)) = to_visit.pop_front() {
            let mut new_edges = Vec::new();
            let mut curr_edge = (u, property);
            loop {
                let (u, property) = curr_edge;
                let Some(v) = edges.remove(&(u, property)) else {
                    break;
                };
                new_edges.push((u, v, property));
                let Some(&(_, new_prop)) = edges.keys().find(|(u, p)| u == &v && valid_successor(property, *p)) else {
                    break
                };
                curr_edge = (v, new_prop);
                add_new_edges(&mut to_visit, v, edges.keys());
            }
            if !new_edges.is_empty() {
                let line = Line::new(u, new_edges);
                lines.push(line);
            }
        }
        edges.is_empty().then(|| LinePattern { nodes, lines })
    }
}

fn add_new_edges<'a, U: Universe + 'a, PEdge: Ord + Copy + 'a>(
    queue: &mut VecDeque<(U, PEdge)>,
    node: U,
    edges: impl IntoIterator<Item = &'a (U, PEdge)>,
) {
    queue.extend(
        edges
            .into_iter()
            .filter(|(u, _)| u == &node)
            .sorted_by_key(|(_, p)| p)
            .copied(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_line_pattern() {
        let mut p: Pattern<usize, (), (usize, usize)> = Default::default();
        p.add_edge(0, 1, (0, 1));
        p.add_edge(1, 2, (1, 0));
        p.add_edge(0, 1, (1, 0));
        p.add_edge(1, 2, (0, 1));
        p.add_edge(0, 1, (2, 2));
        p.add_edge(1, 2, (2, 2));
        let t = |p| {
            if p <= 1 {
                1 - p
            } else {
                p
            }
        };
        assert_eq!(
            p.try_into_line_pattern_with_root(0, |(_, pout), (pin, _)| pin == t(pout)),
            Some(LinePattern {
                nodes: HashMap::new(),
                lines: vec![
                    Line::new(0, vec![(0, 1, (0, 1)), (1, 2, (1, 0)),]),
                    Line::new(0, vec![(0, 1, (1, 0)), (1, 2, (0, 1)),]),
                    Line::new(0, vec![(0, 1, (2, 2)), (1, 2, (2, 2)),])
                ]
            })
        );
    }
}
