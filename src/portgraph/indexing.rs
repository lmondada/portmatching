//! The indexing scheme for port graphs.
//!
//! This binds every node in a port graph to a name, that we use as a
//! key during pattern matching.

use std::{cmp, fmt::Debug, iter};

use portgraph::{LinkView, NodeIndex, PortGraph, PortIndex, PortOffset, PortView};

use crate::{
    indexing::{BindingResult, MissingIndexKeys, ValidBindings},
    HashMap, IndexingScheme,
};

use super::root_candidates::find_root_candidates;

/// The indexing scheme for port graphs.
///
/// The keys are fixed relative to a set of root nodes and a path starting
/// in that root node. Here, a "path" is a very specific type of graph path:
/// at every node that the path goes through, the path must come in through
/// `PortOffset::Incoming(i)` and leave through the opposite port, i.e.
/// `PortOffset::Outgoing(i)`.
///
/// This indexing scheme can thus be thought as the equivalent of the string
/// indexing scheme (where a key is obtained by the distance from the reference
/// point, the first character), except that the key must also specify which
/// one of the reference points (the roots) it refers to.
#[derive(Clone, Default, Debug)]
pub struct PGIndexingScheme;

/// The key for the indexing scheme
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PGIndexKey {
    /// A key for a root node
    PathRoot {
        /// The index of the root node
        index: usize,
    },
    /// A key for a node along a path, given respective to a root node
    AlongPath {
        /// The root node it is referencing to
        path_root: usize,
        /// The start direction of the path at the root
        path_start_port: PortOffset,
        /// The length of the path from the root to this node
        path_length: usize,
    },
}

impl IndexingScheme<PortGraph> for PGIndexingScheme {
    type Map = HashMap<PGIndexKey, NodeIndex>;

    fn valid_bindings(
        &self,
        key: &PGIndexKey,
        known_bindings: &Self::Map,
        data: &PortGraph,
    ) -> BindingResult<Self, PortGraph> {
        match *key {
            PGIndexKey::PathRoot { index } => {
                if index == 0 {
                    return Ok(ValidBindings(data.nodes_iter().collect()));
                }
                let first_unassigned_index = (0..=index).find(|&index| {
                    known_bindings
                        .get(&PGIndexKey::PathRoot { index })
                        .is_none()
                });
                if first_unassigned_index.is_none() {
                    // `key` has already been assigned
                    Ok(ValidBindings(vec![*known_bindings.get(key).unwrap()]))
                } else if first_unassigned_index.unwrap() < index {
                    // Assign a lower root first
                    Err(MissingIndexKeys(vec![PGIndexKey::PathRoot {
                        index: first_unassigned_index.unwrap(),
                    }]))
                } else {
                    let candidates = find_root_candidates(data, known_bindings);
                    Ok(ValidBindings(candidates))
                }
            }
            PGIndexKey::AlongPath {
                path_root,
                path_start_port,
                path_length,
            } => {
                let path_root = PGIndexKey::PathRoot { index: path_root };
                let Some(&root_binding) = known_bindings.get(&path_root) else {
                    return Err(MissingIndexKeys(vec![path_root]));
                };
                let binding = walk_path_nodes(data, root_binding, path_start_port).nth(path_length);
                Ok(ValidBindings(Vec::from_iter(binding)))
            }
        }
    }
}

impl PGIndexKey {
    /// The i-th root key
    pub fn root(index: usize) -> Self {
        PGIndexKey::PathRoot { index }
    }

    fn cmp_key(&self) -> (usize, Option<PortOffset>, usize) {
        match self {
            PGIndexKey::PathRoot { index } => (*index, None, 0),
            PGIndexKey::AlongPath {
                path_root,
                path_start_port,
                path_length,
            } => (*path_root, Some(*path_start_port), *path_length),
        }
    }
}
impl PartialOrd for PGIndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PGIndexKey {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.cmp_key().cmp(&other.cmp_key())
    }
}

impl Debug for PGIndexKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PathRoot { index } => write!(f, "Root({index})"),
            Self::AlongPath {
                path_root,
                path_start_port,
                path_length,
            } => {
                let port = match path_start_port {
                    PortOffset::Incoming(i) => format!("in{i}"),
                    PortOffset::Outgoing(i) => format!("out{i}"),
                };
                write!(f, "Along({path_root}, {port})@{path_length})")
            }
        }
    }
}

/// Traverse a path, returning an iterator over the nodes and ports on the path
///
/// The triples returned are the incoming port at a node, the node and the
/// outgoing port at a node. The first incoming port is always `None`.
///
/// If `start_offset` is a PortOffset::Incoming port, then the path is traversed
/// in reverse, and thus the first port in the triple will be an outgoing port
/// and the last port will be an incoming port.
pub(super) fn walk_path<'g>(
    graph: &'g PortGraph,
    start: NodeIndex,
    start_offset: PortOffset,
) -> impl Iterator<Item = (Option<PortIndex>, NodeIndex, Option<PortIndex>)> + 'g {
    let mut next_port = graph.port_index(start, start_offset);
    iter::once((None, start, next_port)).chain(iter::from_fn(move || {
        let prev_port = graph.port_link(next_port?);
        let curr = graph.port_node(prev_port?).unwrap();
        if curr == start {
            return None;
        }
        let next_port_offset = match graph.port_offset(prev_port?).unwrap() {
            PortOffset::Incoming(offset) => PortOffset::Outgoing(offset),
            PortOffset::Outgoing(offset) => PortOffset::Incoming(offset),
        };
        next_port = graph.port_index(curr, next_port_offset);
        Some((prev_port, curr, next_port))
    }))
}

pub(super) fn walk_path_nodes<'g>(
    graph: &'g PortGraph,
    start: NodeIndex,
    start_offset: PortOffset,
) -> impl Iterator<Item = NodeIndex> + 'g {
    walk_path(graph, start, start_offset).map(|(_, node, _)| node)
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use itertools::Itertools;
    use portgraph::{LinkMut, PortMut};
    use rstest::rstest;
    use std::hash::Hash;

    use crate::HashMap;

    use super::*;

    #[test]
    fn test_walk_path() {
        let mut graph = PortGraph::new();
        let nodes = (0..4).map(|_| graph.add_node(1, 1)).collect_vec();
        nodes.iter().tuple_windows().for_each(|(&src, &dst)| {
            graph.link_nodes(src, 0, dst, 0).unwrap();
        });
        let path = walk_path_nodes(&graph, nodes[0], PortOffset::Outgoing(0));
        assert_eq!(path.collect_vec(), nodes);
    }

    fn from_edges<V: Hash + Eq>(edges: &[(V, usize, V, usize)]) -> PortGraph {
        let mut graph = PortGraph::new();
        let mut v_to_nodes = HashMap::default();
        for (src, src_port, dst, dst_port) in edges {
            let src_node = *v_to_nodes
                .entry(src)
                .or_insert_with(|| graph.add_node(0, 0));
            let n_out_ports = graph.num_inputs(src_node).max(src_port + 1);
            graph.set_num_ports(src_node, graph.num_inputs(src_node), n_out_ports, |_, _| {});
            let dst_node = *v_to_nodes
                .entry(dst)
                .or_insert_with(|| graph.add_node(0, 0));
            let n_in_ports = graph.num_outputs(dst_node).max(dst_port + 1);
            graph.set_num_ports(dst_node, n_in_ports, graph.num_outputs(dst_node), |_, _| {});
            graph
                .link_nodes(src_node, *src_port, dst_node, *dst_port)
                .unwrap();
        }
        graph
    }

    /// Assign all possible bindings to `graph`
    fn all_bindings(graph: &PortGraph) -> Vec<HashMap<PGIndexKey, NodeIndex>> {
        let mut final_bindings = vec![];
        let mut curr_bindings = vec![(0, HashMap::default())];

        while let Some((root_index, bindings)) = curr_bindings.pop() {
            let root_key = PGIndexKey::PathRoot { index: root_index };
            let options = PGIndexingScheme.valid_bindings(&root_key, &bindings, graph);
            let Ok(ValidBindings(candidates)) = options else {
                panic!("Could not assign bindings for root {root_index}");
            };
            if candidates.is_empty() {
                final_bindings.push(bindings);
                continue;
            }
            for root_binding in candidates {
                let mut new_bindings = bindings.clone();
                new_bindings.insert(root_key, root_binding);
                for port in graph.all_port_offsets(root_binding) {
                    let path_len = walk_path_nodes(graph, root_binding, port).count();
                    for path_length in 1..path_len {
                        let key = PGIndexKey::AlongPath {
                            path_root: root_index,
                            path_start_port: port,
                            path_length,
                        };
                        let Ok(ValidBindings(candidates)) =
                            PGIndexingScheme.valid_bindings(&key, &new_bindings, graph)
                        else {
                            panic!("Could not assign bindings for key {key:?}");
                        };
                        let binding = candidates.into_iter().exactly_one().unwrap();
                        new_bindings.insert(key, binding);
                    }
                }
                curr_bindings.push((root_index + 1, new_bindings));
            }
        }
        final_bindings
    }

    #[rstest]
    #[case(1, vec![(0, 0, 1, 0), (1, 0, 2, 0), (2, 0, 3, 0)])]
    #[case(2, vec![(0, 0, 1, 0), (0, 1, 2, 0), (3, 0, 0, 0), (3, 1, 4, 0)])]
    #[case(3, vec![(0, 0, 1, 0), (2, 0, 0, 0), (1, 0, 2, 0)])]
    #[case(4, vec![(0, 0, 1, 0), (2, 0, 0, 0), (2, 1, 1, 1)])]
    fn test_assign_bindings(
        #[case] case_id: usize,
        #[case] edges: Vec<(usize, usize, usize, usize)>,
    ) {
        let graph = from_edges(&edges);
        assert_debug_snapshot!(format!("assign_bindings_{case_id}"), all_bindings(&graph));
    }
}
