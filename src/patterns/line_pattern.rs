use std::collections::VecDeque;

use derive_more::{From, Into};

use crate::{
    predicate::{EdgePredicate, NodeLocation, Symbol, SymbolsIter},
    EdgeProperty, HashMap, NodeProperty, Universe,
};

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
pub struct LinePattern<U: Universe, PNode, PEdge> {
    pub(crate) nodes: HashMap<U, PNode>,
    pub(crate) lines: Vec<Line<U, PEdge>>,
}

/// Within a line pattern, the address of a node
/// is the index of the line and the index of the node within the line
#[derive(Clone, Copy, From, Into, Debug)]
struct LineAddress(usize, usize);

impl<U: Universe, PNode: NodeProperty, PEdge: Clone> LinePattern<U, PNode, PEdge> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn require(&mut self, node: U, property: PNode) {
        self.nodes.insert(node, property);
    }

    pub fn add_line(&mut self, root: U, edges: Vec<(U, U, PEdge)>) {
        self.lines.push(Line { root, edges });
    }

    pub(crate) fn edge_predicates(&self) -> PredicatesIter<'_, U, PNode, PEdge>
    where
        PEdge: EdgeProperty,
    {
        PredicatesIter::new(self)
    }

    fn n_lines(&self) -> usize {
        self.lines.len()
    }
}

impl<U: Universe, PNode, PEdge> Default for LinePattern<U, PNode, PEdge> {
    fn default() -> Self {
        Self {
            nodes: HashMap::default(),
            lines: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PredicatesIter<'a, U: Universe, PNode, PEdge: EdgeProperty> {
    // Map each node to the first address where it occurs
    to_line_ind: HashMap<U, LineAddress>,
    // For each line, the first index that we have not yet visited
    // (index 0 is always discovered by some other line)
    visited_boundary: Vec<usize>,
    // The queue of next items
    it_queue: VecDeque<EdgePredicate<PNode, PEdge, PEdge::OffsetID>>,
    // The symbols that have been assigned to each node
    u_to_symbols: HashMap<U, Symbol>,
    // How far we have come in the traversal
    status: IterationStatus,
    // The node weights
    nodes: &'a HashMap<U, PNode>,
    // The edges
    lines: &'a Vec<Line<U, PEdge>>,
    // The next free symbols to use
    free_symbols: SymbolsIter,
}

impl<'a, U: Universe, PNode: NodeProperty, PEdge: EdgeProperty>
    PredicatesIter<'a, U, PNode, PEdge>
{
    fn new(p: &'a LinePattern<U, PNode, PEdge>) -> Self {
        let lines = &p.lines;
        let nodes = &p.nodes;
        let mut to_line_ind = HashMap::default();
        let mut it_queue = VecDeque::new();
        let mut u_to_symbols = HashMap::default();
        let mut status = IterationStatus::Finished;
        let mut free_symbols = Symbol::symbols_in_status(status);
        if let Some(root) = lines.get(0).map(|w| w.root) {
            to_line_ind.insert(root, (0, 0).into());
            // Add 0-th line to known nodes
            for (ind, &(_, u, _)) in lines[0].edges.iter().enumerate() {
                to_line_ind.entry(u).or_insert_with(|| (0, ind + 1).into());
            }
            status = IterationStatus::Skeleton(0);
            free_symbols = Symbol::symbols_in_status(status);
            let root_symbol = free_symbols.next().unwrap();
            u_to_symbols.insert(root, root_symbol);
            if let Some(root_prop) = nodes.get(&root) {
                it_queue.push_back(EdgePredicate::NodeProperty {
                    node: root_symbol,
                    property: root_prop.clone(),
                });
            }
            if let Some(first_prop) = lines[0].edges.get(0).map(|w| &w.2) {
                // Indicate location of first line
                it_queue.push_back(EdgePredicate::NextRoot {
                    line_nb: 0,
                    new_root: NodeLocation::Exists(root_symbol),
                    offset: first_prop.offset_id(),
                });
            }
        }
        Self {
            to_line_ind,
            visited_boundary: vec![1; p.n_lines()],
            it_queue,
            u_to_symbols,
            status,
            lines,
            nodes,
            free_symbols,
        }
    }

    pub(crate) fn peek(&mut self) -> Option<&EdgePredicate<PNode, PEdge, PEdge::OffsetID>> {
        self.fill_queue();
        self.it_queue.front()
    }

    pub(crate) fn traversal_stage(&mut self) -> IterationStatus {
        self.fill_queue();
        self.status
    }

    fn n_lines(&self) -> usize {
        self.lines.len()
    }

    fn reach_ith_root(&mut self, i: usize, first_prop: &PEdge)
    where
        PEdge: EdgeProperty,
    {
        // Follow the path from a visited node to the root of line i
        let (j, j_ind) = self.to_line_ind[&self.lines[i].root].into();
        let boundary_ind = &mut self.visited_boundary[j];
        if j_ind >= *boundary_ind {
            // Indicate that we will be traversing `j` to reach the root of `i`
            self.it_queue.push_back(EdgePredicate::NextRoot {
                line_nb: i,
                new_root: NodeLocation::Discover(j),
                offset: first_prop.offset_id(),
            });
        } else {
            // Indicate that the root of `i` is known
            self.it_queue.push_back(EdgePredicate::NextRoot {
                line_nb: i,
                new_root: NodeLocation::Exists(self.u_to_symbols[&self.lines[i].root]),
                offset: first_prop.offset_id(),
            });
        }
        while j_ind >= *boundary_ind {
            self.it_queue.extend(edge_predicates(
                self.lines[j].edges[*boundary_ind - 1].clone(),
                &mut self.u_to_symbols,
                self.nodes,
                self.free_symbols.by_ref(),
            ));
            *boundary_ind += 1;
        }
    }

    fn traverse_leftover(&mut self, i: usize) {
        if i == 0 {
            // Indicate that we are moving to left overs
            self.it_queue.push_back(EdgePredicate::True);
        }
        for ind in (self.visited_boundary[i] - 1)..self.lines[i].edges.len() {
            self.it_queue.extend(edge_predicates(
                self.lines[i].edges[ind].clone(),
                &mut self.u_to_symbols,
                self.nodes,
                self.free_symbols.by_ref(),
            ));
        }
        self.visited_boundary[i] = self.lines[i].edges.len() + 1;
    }

    fn fill_queue(&mut self) {
        while self.it_queue.is_empty() {
            // Increment status
            self.status.increment(self.n_lines());
            // Update free symbols to match new status
            self.free_symbols = Symbol::symbols_in_status(self.status);

            match self.status {
                IterationStatus::Skeleton(i) => {
                    // Append predicates to reach the root of the i-th line
                    self.reach_ith_root(
                        i,
                        &self.lines[i]
                            .edges
                            .first()
                            .expect("Cannot match empty line")
                            .2,
                    );

                    // Add i-th line to known nodes
                    for (ind, &(_, u, _)) in self.lines[i].edges.iter().enumerate() {
                        self.to_line_ind
                            .entry(u)
                            .or_insert_with(|| (i, ind + 1).into());
                    }
                }
                IterationStatus::LeftOver(i) => {
                    self.traverse_leftover(i);
                }
                IterationStatus::Finished => break,
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
}

impl<'a, U, PNode, PEdge> Iterator for PredicatesIter<'a, U, PNode, PEdge>
where
    U: Universe,
    PNode: NodeProperty,
    PEdge: EdgeProperty,
{
    type Item = EdgePredicate<PNode, PEdge, PEdge::OffsetID>;

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

fn edge_predicates<U: Universe, PNode: NodeProperty, PEdge: EdgeProperty>(
    (u, v, property): (U, U, PEdge),
    symbols: &mut HashMap<U, Symbol>,
    nodes: &HashMap<U, PNode>,
    mut new_symbols: impl Iterator<Item = Symbol>,
) -> Vec<EdgePredicate<PNode, PEdge, PEdge::OffsetID>> {
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
        if let Some(v_prop) = nodes.get(&v).cloned() {
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
        let symbs = [
            Symbol::new(IterationStatus::Skeleton(0), 0),
            Symbol::new(IterationStatus::Skeleton(1), 0),
            Symbol::new(IterationStatus::Skeleton(1), 1),
            Symbol::new(IterationStatus::Skeleton(2), 0),
            Symbol::new(IterationStatus::Skeleton(2), 1),
            Symbol::new(IterationStatus::LeftOver(2), 0),
            Symbol::new(IterationStatus::LeftOver(2), 1),
        ];
        assert_eq!(
            p.edge_predicates().collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: symbs[0],
                    property: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 0,
                    new_root: NodeLocation::Exists(symbs[0]),
                    offset: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 1,
                    new_root: NodeLocation::Discover(0),
                    offset: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: (),
                    new_node: symbs[1]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[1],
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: (),
                    new_node: symbs[2]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[2],
                    property: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 2,
                    new_root: NodeLocation::Discover(1),
                    offset: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[2],
                    property: (),
                    new_node: symbs[3]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[3],
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[3],
                    property: (),
                    new_node: symbs[4]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[4],
                    property: ()
                },
                EdgePredicate::True,
                EdgePredicate::LinkKnownNode {
                    node: symbs[2],
                    property: (),
                    known_node: symbs[3]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[4],
                    property: (),
                    new_node: symbs[5]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[5],
                    property: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[5],
                    property: (),
                    new_node: symbs[6]
                },
                EdgePredicate::NodeProperty {
                    node: symbs[6],
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
        let symbs = [
            Symbol::new(IterationStatus::Skeleton(0), 0),
            Symbol::new(IterationStatus::Skeleton(2), 0),
            Symbol::new(IterationStatus::LeftOver(0), 0),
            Symbol::new(IterationStatus::LeftOver(0), 1),
            Symbol::new(IterationStatus::LeftOver(1), 0),
            Symbol::new(IterationStatus::LeftOver(2), 0),
        ];
        assert_eq!(
            p2.edge_predicates().collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: symbs[0],
                    property: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 0,
                    new_root: NodeLocation::Exists(Symbol::new(IterationStatus::Skeleton(0), 0)),
                    offset: 0
                },
                EdgePredicate::NextRoot {
                    line_nb: 1,
                    new_root: NodeLocation::Exists(Symbol::new(IterationStatus::Skeleton(0), 0)),
                    offset: 0
                },
                EdgePredicate::NextRoot {
                    line_nb: 2,
                    new_root: NodeLocation::Discover(1),
                    offset: 0
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: 2,
                    new_node: symbs[1]
                },
                EdgePredicate::True,
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: 0,
                    new_node: symbs[2]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[2],
                    property: 1,
                    new_node: symbs[3]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: 3,
                    new_node: symbs[4]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: 4,
                    new_node: symbs[5]
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

        let symbs = [
            Symbol::new(IterationStatus::Skeleton(0), 0),
            Symbol::new(IterationStatus::Skeleton(1), 0),
            Symbol::new(IterationStatus::Skeleton(1), 1),
            Symbol::new(IterationStatus::LeftOver(1), 0),
        ];
        assert_eq!(
            p2.edge_predicates().collect::<Vec<_>>(),
            vec![
                EdgePredicate::NodeProperty {
                    node: symbs[0],
                    property: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 0,
                    new_root: NodeLocation::Exists(symbs[0]),
                    offset: 0
                },
                EdgePredicate::NextRoot {
                    line_nb: 1,
                    new_root: NodeLocation::Discover(0),
                    offset: 0
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: 0,
                    new_node: symbs[1]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: 0,
                    new_node: symbs[2]
                },
                EdgePredicate::True,
                EdgePredicate::LinkNewNode {
                    node: symbs[2],
                    property: 0,
                    new_node: symbs[3]
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
        let symbs = [
            Symbol::new(IterationStatus::Skeleton(0), 0),
            Symbol::new(IterationStatus::Skeleton(2), 0),
            Symbol::new(IterationStatus::LeftOver(0), 0),
            Symbol::new(IterationStatus::LeftOver(0), 1),
            Symbol::new(IterationStatus::LeftOver(1), 0),
            Symbol::new(IterationStatus::LeftOver(2), 0),
        ];
        assert_eq!(
            p2.edge_predicates().collect_vec(),
            vec![
                EdgePredicate::NodeProperty {
                    node: symbs[0],
                    property: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 0,
                    new_root: NodeLocation::Exists(Symbol::new(IterationStatus::Skeleton(0), 0)),
                    offset: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 1,
                    new_root: NodeLocation::Exists(Symbol::new(IterationStatus::Skeleton(0), 0)),
                    offset: ()
                },
                EdgePredicate::NextRoot {
                    line_nb: 2,
                    new_root: NodeLocation::Discover(1),
                    offset: ()
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: (),
                    new_node: symbs[1]
                },
                EdgePredicate::True,
                EdgePredicate::LinkNewNode {
                    node: symbs[0],
                    property: (),
                    new_node: symbs[2]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[2],
                    property: (),
                    new_node: symbs[3]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: (),
                    new_node: symbs[4]
                },
                EdgePredicate::LinkNewNode {
                    node: symbs[1],
                    property: (),
                    new_node: symbs[5]
                },
            ]
        )
    }
}
