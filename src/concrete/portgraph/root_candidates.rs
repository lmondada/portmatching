//! Logic to find potential root assignments given a port graph and known bindings.
//!
//! This can be done fairly arbirarily, as long as it is consistent. This whole
//! module is hilariously inefficient, we will see when this becomes the bottleneck.
//!
//! ## Main idea
//! Every time a new root should be assign, we build a spanning tree of all
//! the previous roots that have been assigned. We can then figure out where
//! the spanning tree can be extended to add new roots.

use itertools::Itertools;
use portgraph::{NodeIndex, PortGraph, PortIndex, PortOffset, PortView};

use crate::{HashMap, HashSet};

use super::indexing::{walk_path, PGIndexKey};

pub(super) fn find_root_candidates(
    graph: &PortGraph,
    bindings: &HashMap<PGIndexKey, NodeIndex>,
) -> Vec<NodeIndex> {
    let tree = RootSpanningTree::new(graph, bindings);

    let mut root_candidates = Vec::new();

    for root in tree.nodes {
        // For each root, we can use the paths leaving ports larger than any
        // we have used for previous roots (rationale: otherwise we'd have
        // chosen that root first)
        let max_offset_used = root
            .neighbours
            .iter()
            .filter(|(_, v)| matches!(v, NeighbourType::KnownRoot(_)))
            .map(|(&offset, _)| offset)
            .max();
        root_candidates.extend(root.neighbours.into_iter().filter_map(|(k, v)| {
            if Some(k) <= max_offset_used {
                return None;
            }
            if let NeighbourType::NewRoot(new_root) = v {
                return Some(new_root);
            }
            None
        }));
    }
    root_candidates
}

#[derive(Clone, Debug, Default)]
struct RootSpanningNode {
    neighbours: HashMap<PortOffset, NeighbourType>,
}

#[derive(Clone, Debug)]
enum NeighbourType {
    Parent,
    KnownRoot(usize),
    NewRoot(NodeIndex),
}

/// A spanning tree of all the assigned roots in the graph.
///
/// At each node store where following the path along each of its ports leads
/// to. It may lead to either i) the root we come from, ii) another assigned
/// root or iii) a node with free ports that we can extend the tree at.
///
/// We ignore back-edges, i.e. paths that lead to a root already in the spanning tree.
#[derive(Clone, Debug)]
struct RootSpanningTree {
    nodes: Vec<RootSpanningNode>,
}

impl RootSpanningTree {
    fn new(graph: &PortGraph, bindings: &HashMap<PGIndexKey, NodeIndex>) -> Self {
        let known_roots = (0..)
            .map(|i| bindings.get(&PGIndexKey::PathRoot { index: i }).map(|v| *v))
            .while_some()
            .collect_vec();
        let known_roots_inv: HashMap<_, _> = known_roots.iter().copied().zip(0..).collect();
        let known_nodes = bindings.iter().map(|(_, &v)| v).collect_vec();
        let nodes_with_free_ports = nodes_with_free_ports(graph, bindings);

        let mut tree_nodes = vec![RootSpanningNode::default(); known_roots.len()];
        let mut seen_roots = HashSet::default();
        for (i, &node) in known_roots.iter().enumerate() {
            seen_roots.insert(i);
            for port in graph.all_port_offsets(node) {
                let neighbour_type = traverse_path_neighbour_type(
                    graph,
                    node,
                    port,
                    i,
                    &mut seen_roots,
                    &known_nodes,
                    &known_roots_inv,
                    &nodes_with_free_ports,
                );
                if let Some(neighbour_type) = neighbour_type {
                    tree_nodes[i].neighbours.insert(port, neighbour_type);
                }
            }
        }
        Self { nodes: tree_nodes }
    }
}

/// Traverse the path starting in (`node`, `port`) and return the type of
/// root found on the path.
///
/// More precisely:
///  - if the path leads to a smaller root, return `NeighbourType::Parent`.
///  - if the path leads to a known, larger root, return `NeighbourType::KnownRoot`
///  - otherwise, return the first potential new root along the path as
///    `NeighbourType::NewRoot`.
///
/// We handle path cycle by picking the direction of traversal that starts in a
/// smaller offset as "finding a larger root", and the other direction as "finding
/// a smaller root".
fn traverse_path_neighbour_type(
    graph: &PortGraph,
    node: NodeIndex,
    port: PortOffset,
    current_root: usize,
    seen_roots: &mut HashSet<usize>,
    known_nodes: &[NodeIndex],
    known_roots_inv: &HashMap<NodeIndex, usize>,
    nodes_with_free_ports: &HashSet<NodeIndex>,
) -> Option<NeighbourType> {
    let mut path = Vec::new();
    for (incoming_p, node, _) in walk_path(graph, node, port).skip(1) {
        if !known_nodes.contains(&node) {
            break;
        }
        let incoming_offset = incoming_p.map(|p| graph.port_offset(p).unwrap());
        if let Some(&root) = known_roots_inv.get(&node) {
            if root < current_root {
                return Some(NeighbourType::Parent);
            } else if root == current_root && Some(port) > incoming_offset {
                return Some(NeighbourType::Parent);
            } else if seen_roots.insert(root) {
                return Some(NeighbourType::KnownRoot(root));
            } else {
                break;
            }
        }
        path.push(node);
    }
    path.into_iter()
        .find(|n| nodes_with_free_ports.contains(n))
        .map(|n| NeighbourType::NewRoot(n))
}

/// For each node the ports that have not been traversed.
fn free_ports(
    graph: &PortGraph,
    bindings: &HashMap<PGIndexKey, NodeIndex>,
) -> HashMap<NodeIndex, Vec<PortIndex>> {
    // Find the paths that have already been traversed
    // Map from (root, root_offset) to the length of the path traversed
    let mut paths = HashMap::<_, usize>::default();
    let mut roots = HashSet::default();
    for (key, node) in bindings.iter() {
        if let PGIndexKey::AlongPath {
            path_root,
            path_start_port,
            path_length,
        } = key
        {
            let path_id = (*path_root, *path_start_port);
            let prev_len = paths.entry(path_id).or_default();
            *prev_len = (*path_length).max(*prev_len);
        } else {
            roots.insert(node);
        }
    }

    // Traverse each of the traversed paths and remove the ports that have been used
    let mut free_ports = HashMap::default();
    for (&(root, root_offset), &length) in paths.iter() {
        let root = PGIndexKey::PathRoot { index: root };
        let &root_node = bindings
            .get(&root)
            .expect("A PGIndexKey::AlongPath binding references an unbound root");
        for (p1, node, p2) in walk_path(graph, root_node, root_offset).take(length + 1) {
            if roots.contains(&node) {
                continue;
            }
            let free_ports = free_ports
                .entry(node)
                .or_insert_with(|| graph.all_ports(node).collect_vec());
            if let Some(p1) = p1.and_then(|p1| free_ports.iter().position(|&p| p == p1)) {
                free_ports.remove(p1);
            }
            if let Some(p2) = p2.and_then(|p2| free_ports.iter().position(|&p| p == p2)) {
                free_ports.remove(p2);
            }
        }
    }
    free_ports
}

/// All bound nodes that have ports that have not been traversed yet.
///
/// TODO: This is outrageously inefficient.
fn nodes_with_free_ports(
    graph: &PortGraph,
    bindings: &HashMap<PGIndexKey, NodeIndex>,
) -> HashSet<NodeIndex> {
    let free_ports = free_ports(graph, bindings);
    free_ports
        .into_iter()
        .filter(|(_, ports)| !ports.is_empty())
        .map(|(node, _)| node)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use insta::assert_debug_snapshot;
    use portgraph::{LinkMut, PortGraph, PortMut};
    use rstest::{fixture, rstest};

    #[fixture]
    fn small_graph() -> PortGraph {
        let mut graph = PortGraph::new();
        let n0 = graph.add_node(2, 2);
        let n1 = graph.add_node(1, 1);
        let n2 = graph.add_node(1, 2);
        let n3 = graph.add_node(1, 2);

        graph.link_nodes(n0, 0, n1, 0).unwrap();
        graph.link_nodes(n1, 0, n2, 0).unwrap();
        graph.link_nodes(n3, 1, n0, 0).unwrap();

        graph
    }

    fn nodes() -> (NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
        (0..4)
            .map(|i| i.try_into().unwrap())
            .collect_tuple()
            .unwrap()
    }

    #[fixture]
    fn bindings() -> HashMap<PGIndexKey, NodeIndex> {
        let (n0, n1, n2, n3) = nodes();
        HashMap::from_iter([
            (PGIndexKey::PathRoot { index: 0 }, n0),
            (
                PGIndexKey::AlongPath {
                    path_root: 0,
                    path_start_port: PortOffset::Outgoing(0),
                    path_length: 1,
                },
                n1,
            ),
            (
                PGIndexKey::AlongPath {
                    path_root: 0,
                    path_start_port: PortOffset::Outgoing(0),
                    path_length: 2,
                },
                n2,
            ),
            (
                PGIndexKey::AlongPath {
                    path_root: 0,
                    path_start_port: PortOffset::Incoming(0),
                    path_length: 1,
                },
                n3,
            ),
        ])
    }

    #[rstest]
    fn test_free_ports(small_graph: PortGraph, bindings: HashMap<PGIndexKey, NodeIndex>) {
        let free_ports = free_ports(&small_graph, &bindings);
        let (_, n1, n2, n3) = nodes();
        let exp_free_ports: HashMap<_, Vec<_>> = [
            (n2, vec![PortOffset::Outgoing(1)]),
            (n3, vec![PortOffset::Incoming(0), PortOffset::Outgoing(0)]),
            (n1, vec![]),
        ]
        .into_iter()
        .map(|(n, offsets)| {
            (
                n,
                offsets
                    .into_iter()
                    .map(|offset| small_graph.port_index(n, offset).unwrap())
                    .collect_vec(),
            )
        })
        .collect();
        assert_eq!(free_ports, exp_free_ports);
    }

    #[rstest]
    fn test_find_root_candidates(small_graph: PortGraph, bindings: HashMap<PGIndexKey, NodeIndex>) {
        let (_, _, n2, n3) = (0..4)
            .map(|i| i.try_into().unwrap())
            .collect_tuple()
            .unwrap();
        let candidates = find_root_candidates(&small_graph, &bindings);
        assert_eq!(HashSet::from_iter(candidates), HashSet::from_iter([n2, n3]));
    }

    #[rstest]
    fn test_spanning_tree(small_graph: PortGraph, mut bindings: HashMap<PGIndexKey, NodeIndex>) {
        let tree = RootSpanningTree::new(&small_graph, &bindings);
        assert_debug_snapshot!("RootSpanningTree_one_root", tree);
        bindings.insert(PGIndexKey::PathRoot { index: 1 }, nodes().2);
        let tree = RootSpanningTree::new(&small_graph, &bindings);
        assert_debug_snapshot!("RootSpanningTree_two_roots", tree);
    }
}
