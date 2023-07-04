use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    collections::{HashMap, VecDeque},
    default,
    iter::Map,
    mem,
    ops::RangeFrom,
};

use derive_more::{From, Into};

use crate::{predicate::EdgePredicate, Universe};

#[derive(PartialEq, Eq, Clone, Debug)]
pub(crate) struct Line<U, PEdge> {
    root: U,
    edges: Vec<(U, U, PEdge)>,
}

impl<U, PEdge> Line<U, PEdge> {
    pub(crate) fn new(root: U, edges: Vec<(U, U, PEdge)>) -> Self {
        Self { root, edges }
    }
}

/// A pattern to match, stored line by line from the root
#[derive(PartialEq, Eq, Clone, Debug)]
pub(crate) struct LinePattern<U: Universe, PNode, PEdge> {
    pub(crate) nodes: HashMap<U, PNode>,
    pub(crate) lines: Vec<Line<U, PEdge>>,
}

/// Within a line pattern, the address of a node
/// is the index of the line and the index of the node within the line
#[derive(Clone, Copy, From, Into, Debug)]
struct LineAddress(usize, usize);

impl<U: Universe, PNode: Copy, PEdge: Copy> LinePattern<U, PNode, PEdge> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn require(&mut self, node: U, property: PNode) {
        self.nodes.insert(node, property);
    }

    pub fn add_line(&mut self, root: U, edges: Vec<(U, U, PEdge)>) {
        self.lines.push(Line { root, edges });
    }

    pub(crate) fn edge_predicates<IS: Iterator>(
        &self,
        free_symbols: IS,
    ) -> PredicatesIter<'_, U, PNode, PEdge, IS, IS::Item>
    where
        IS::Item: Copy,
    {
        PredicatesIter::new(self, free_symbols)
    }

    fn n_lines(&self) -> usize {
        self.lines.len()
    }

    fn n_edges(&self) -> usize {
        self.lines.iter().map(|l| l.edges.len()).sum()
    }

    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

impl<U: Universe, PNode, PEdge> Default for LinePattern<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            lines: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PredicatesIter<'a, U: Universe, PNode, PEdge, IS, S> {
    // Map each node to the first address where it occurs
    to_line_ind: HashMap<U, LineAddress>,
    // For each line, the first index that we have not yet visited
    // (index 0 is always discovered by some other line)
    visited_boundary: Vec<usize>,
    // The queue of next items
    it_queue: VecDeque<EdgePredicate<PNode, PEdge, S>>,
    // The symbols that have been assigned to each node
    u_to_symbols: HashMap<U, S>,
    // An infinite supply of new symbols
    free_symbols: IS,
    // How far we have come in the traversal
    status: IterationStatus,
    // The node weights
    nodes: &'a HashMap<U, PNode>,
    // The edges
    lines: &'a Vec<Line<U, PEdge>>,
}

impl<'a, U: Universe, PNode: Copy, PEdge: Copy, IS, S: Copy>
    PredicatesIter<'a, U, PNode, PEdge, IS, S>
{
    fn new(p: &'a LinePattern<U, PNode, PEdge>, mut free_symbols: IS) -> Self
    where
        IS: Iterator<Item = S>,
    {
        let lines = &p.lines;
        let nodes = &p.nodes;
        let mut to_line_ind = HashMap::new();
        let mut it_queue = VecDeque::new();
        let mut u_to_symbols = HashMap::new();
        let mut status = IterationStatus::Finished;
        if let Some(root) = lines.get(0).map(|w| w.root) {
            to_line_ind.insert(root, (0, 0).into());
            // Add 0-th line to known nodes
            for (ind, &(_, u, _)) in lines[0].edges.iter().enumerate() {
                if !to_line_ind.contains_key(&u) {
                    to_line_ind.insert(u, (0, ind + 1).into());
                }
            }
            let root_symbol = free_symbols.next().expect("Could not get new symbol");
            u_to_symbols.insert(root, root_symbol);
            status = IterationStatus::Skeleton(0);
            if let Some(&root_prop) = nodes.get(&root) {
                it_queue.push_back(EdgePredicate::NodeProperty {
                    node: root_symbol,
                    property: root_prop,
                });
            }
        }
        Self {
            to_line_ind,
            visited_boundary: vec![1; p.n_lines()],
            it_queue,
            u_to_symbols,
            free_symbols,
            status,
            lines,
            nodes,
        }
    }

    pub(crate) fn peek(&mut self) -> Option<EdgePredicate<PNode, PEdge, S>>
    where
        IS: Iterator<Item = S>,
    {
        self.fill_queue();
        self.it_queue.front().copied()
    }

    pub(crate) fn traversal_stage(&mut self) -> IterationStatus
    where
        IS: Iterator<Item = S>,
    {
        self.fill_queue();
        self.status
    }

    fn n_lines(&self) -> usize {
        self.lines.len()
    }

    fn reach_ith_root(&mut self, i: usize)
    where
        IS: Iterator<Item = S>,
    {
        // Follow the path from a visited node to the root of line i
        let (j, j_ind) = self.to_line_ind[&self.lines[i].root].into();
        let boundary_ind = &mut self.visited_boundary[j];
        // if j_ind >= *boundary_ind {
        //     // Indicate that we will be traversing `j`
        //     self.it_queue
        //         .push_back(EdgePredicate::TraverseAlong { line: j });
        // }
        while j_ind >= *boundary_ind {
            self.it_queue.extend(edge_predicates(
                self.lines[j].edges[*boundary_ind - 1],
                &mut self.u_to_symbols,
                self.nodes,
                self.free_symbols.by_ref(),
            ));
            *boundary_ind += 1;
        }
    }

    fn traverse_leftover(&mut self, i: usize)
    where
        IS: Iterator<Item = S>,
    {
        for ind in (self.visited_boundary[i] - 1)..self.lines[i].edges.len() {
            self.it_queue.extend(edge_predicates(
                self.lines[i].edges[ind],
                &mut self.u_to_symbols,
                self.nodes,
                self.free_symbols.by_ref(),
            ));
        }
        self.visited_boundary[i] = self.lines[i].edges.len() + 1;
    }

    fn fill_queue(&mut self)
    where
        IS: Iterator<Item = S>,
    {
        while self.it_queue.is_empty() {
            // Increment status
            self.status.increment(self.n_lines());

            match self.status {
                IterationStatus::Skeleton(i) => {
                    if i > 0 {
                        // Append predicates to reach the root of the i-th line
                        self.reach_ith_root(i);
                    }

                    // Add i-th line to known nodes
                    for (ind, &(_, u, _)) in self.lines[i].edges.iter().enumerate() {
                        if !self.to_line_ind.contains_key(&u) {
                            self.to_line_ind.insert(u, (i, ind + 1).into());
                        }
                    }
                }
                IterationStatus::LeftOver(i) => {
                    self.traverse_leftover(i);
                }
                IterationStatus::Finished => break,
            }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum IterationStatus {
    // We are traversing the i-th line wihtin skeleton
    Skeleton(usize),
    // We are traversing the i-th line outside skeleton
    LeftOver(usize),
    // We are done
    Finished,
}

impl IterationStatus {
    fn increment(&mut self, max_i: usize) {
        *self = match self {
            Self::Skeleton(i) => {
                if *i + 1 < max_i {
                    Self::Skeleton(*i + 1)
                } else {
                    Self::LeftOver(0)
                }
            }
            Self::LeftOver(i) => {
                if *i + 1 < max_i {
                    Self::LeftOver(*i + 1)
                } else {
                    Self::Finished
                }
            }
            Self::Finished => Self::Finished,
        }
    }

    pub(super) fn leftover_index(&self) -> Option<usize> {
        match self {
            Self::LeftOver(i) => Some(*i),
            _ => None,
        }
    }
}

impl<'a, U, PNode, PEdge, IS> Iterator for PredicatesIter<'a, U, PNode, PEdge, IS, IS::Item>
where
    U: Universe,
    PNode: Copy,
    PEdge: Copy,
    IS: Iterator,
    IS::Item: Copy,
{
    type Item = EdgePredicate<PNode, PEdge, IS::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.to_line_ind.is_empty() {
            return None;
        }

        if let Some(val) = self.it_queue.pop_front() {
            return Some(val);
        }

        self.fill_queue();

        self.it_queue.pop_front()
    }
}

fn edge_predicates<U: Universe, PNode: Copy, PEdge: Copy, IS: Iterator>(
    (u, v, property): (U, U, PEdge),
    symbols: &mut HashMap<U, IS::Item>,
    nodes: &HashMap<U, PNode>,
    mut new_symbols: IS,
) -> Vec<EdgePredicate<PNode, PEdge, IS::Item>>
where
    IS::Item: Copy,
{
    let mut preds = Vec::new();

    let u_symb = symbols[&u];
    if let Some(&v_symb) = symbols.get(&v) {
        preds.push(EdgePredicate::LinkKnownNode {
            node: u_symb,
            property,
            known_node: v_symb,
        });
    } else {
        let v_symb = new_symbols.next().unwrap();
        preds.push(EdgePredicate::LinkNewNode {
            node: u_symb,
            property,
            new_node: v_symb,
        });
        if let Some(&v_prop) = nodes.get(&v) {
            preds.push(EdgePredicate::NodeProperty {
                node: v_symb,
                property: v_prop,
            });
        }
        symbols.insert(v, v_symb);
    }
    preds
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_predicate_creation() {
        let l1 = Line {
            root: 0,
            edges: vec![(0, 1, ()), (1, 3, ()), (3, 6, ())],
        };
        let l2 = Line {
            root: 3,
            edges: vec![(3, 6, ()), (6, 7, ())],
        };
        let l3 = Line {
            root: 7,
            edges: vec![(7, 8, ()), (8, 9, ())],
        };
        let p = LinePattern {
            nodes: HashMap::from_iter([
                (0, ()),
                (1, ()),
                (3, ()),
                (6, ()),
                (7, ()),
                (8, ()),
                (9, ()),
            ]),
            lines: vec![l1, l2, l3],
        };
        assert_eq!(
            p.edge_predicates(0..).collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: 0,
                    property: ()
                },
                // EdgePredicate::TraverseAlong { line: 0 },
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: (),
                    new_node: 1
                },
                EdgePredicate::NodeProperty {
                    node: 1,
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: 1,
                    property: (),
                    new_node: 2
                },
                EdgePredicate::NodeProperty {
                    node: 2,
                    property: ()
                },
                // EdgePredicate::TraverseAlong { line: 1 },
                EdgePredicate::LinkNewNode {
                    node: 2,
                    property: (),
                    new_node: 3
                },
                EdgePredicate::NodeProperty {
                    node: 3,
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: 3,
                    property: (),
                    new_node: 4
                },
                EdgePredicate::NodeProperty {
                    node: 4,
                    property: ()
                },
                // TODO: the deterministic part from here on
                // EdgePredicate::TraverseAlong { line: 2 },
                EdgePredicate::LinkKnownNode {
                    node: 2,
                    property: (),
                    known_node: 3
                },
                EdgePredicate::LinkNewNode {
                    node: 4,
                    property: (),
                    new_node: 5
                },
                EdgePredicate::NodeProperty {
                    node: 5,
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: 5,
                    property: (),
                    new_node: 6
                },
                EdgePredicate::NodeProperty {
                    node: 6,
                    property: ()
                },
            ]
        )
    }

    #[test]
    fn test_to_edges() {
        let mut p2 = LinePattern::new();
        p2.require(0, ());
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 1)]);
        p2.add_line(0, vec![(0, 3, 2), (3, 4, 3)]);
        p2.add_line(3, vec![(3, 5, 4)]);

        // 0 -> Symb(R)
        // 3 -> Symb(1)
        // 1 -> Symb(2)
        // 2 -> Symb(3)
        // 4 -> Symb(4)
        // 5 -> Symb(5)
        assert_eq!(
            p2.edge_predicates(0..).collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: 0,
                    property: ()
                },
                // EdgePredicate::TraverseAlong { line: 1 },
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: 2,
                    new_node: 1.into()
                },
                // EdgePredicate::True,
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: 0,
                    new_node: 2.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 2.into(),
                    property: 1,
                    new_node: 3.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 1.into(),
                    property: 3,
                    new_node: 4.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 1.into(),
                    property: 4,
                    new_node: 5.into()
                },
            ]
        )
    }

    #[test]
    fn test_to_edges_off_by_one() {
        let mut p2 = LinePattern::new();
        p2.require(0, ());
        p2.add_line(0, vec![(0, 1, 0), (1, 2, 0)]);
        p2.add_line(2, vec![(2, 3, 0)]);

        assert_eq!(
            p2.edge_predicates(0..).collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: 0,
                    property: ()
                },
                // EdgePredicate::TraverseAlong { line: 0 },
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: 0,
                    new_node: 1.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 1.into(),
                    property: 0,
                    new_node: 2.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 2.into(),
                    property: 0,
                    new_node: 3.into()
                },
            ]
        )
    }

    #[test]
    fn test_to_edges_no_prop() {
        // The same as above, but now edges have no property
        let mut p2 = LinePattern::new();
        p2.require(0, ());
        p2.add_line(0, vec![(0, 1, ()), (1, 2, ())]);
        p2.add_line(0, vec![(0, 3, ()), (3, 4, ())]);
        p2.add_line(3, vec![(3, 5, ())]);

        // 0 -> Symb(R)
        // 3 -> Symb(1)
        // 1 -> Symb(2)
        // 2 -> Symb(3)
        // 4 -> Symb(4)
        // 5 -> Symb(5)
        assert_eq!(
            p2.edge_predicates(0..).collect_vec(),
            vec![
                EdgePredicate::NodeProperty {
                    node: 0,
                    property: ()
                },
                // EdgePredicate::TraverseAlong { line: 1 },
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: (),
                    new_node: 1.into()
                },
                // EdgePredicate::True,
                EdgePredicate::LinkNewNode {
                    node: 0,
                    property: (),
                    new_node: 2.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 2.into(),
                    property: (),
                    new_node: 3.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 1.into(),
                    property: (),
                    new_node: 4.into()
                },
                EdgePredicate::LinkNewNode {
                    node: 1.into(),
                    property: (),
                    new_node: 5.into()
                },
            ]
        )
    }
}
