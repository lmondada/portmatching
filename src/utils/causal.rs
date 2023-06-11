use std::collections::BTreeSet;

use portgraph::{NodeIndex, PortGraph, SecondaryMap};

/// Returns true if the given node is an ancestor of the given node.
pub(crate) fn is_ancestor<Map: SecondaryMap<NodeIndex, usize>>(
    ancestor: NodeIndex,
    descendant: NodeIndex,
    graph: &PortGraph,
    topsort: &Map,
) -> bool {
    if topsort.get(ancestor) >= topsort.get(descendant) {
        return false;
    }
    let mut ancestors = BTreeSet::from_iter([ancestor]);
    let mut descendants = BTreeSet::from_iter([descendant]);
    while !ancestors.is_empty() && !descendants.is_empty() {
        let mut new_ancestors = BTreeSet::new();
        let descendant_inds = descendants
            .iter()
            .map(|&n| *topsort.get(n))
            .collect::<BTreeSet<_>>();
        for &ancestor in &ancestors {
            for port in graph.output_links(ancestor).flatten() {
                let node = graph.port_node(port).expect("invalid port");
                match topsort.get(node) {
                    ind if descendant_inds.contains(ind) => return true,
                    ind if descendant_inds.last().unwrap() > ind => {
                        new_ancestors.insert(node);
                    }
                    _ => (),
                };
            }
        }
        ancestors = new_ancestors;
        if ancestors.is_empty() {
            break;
        }

        let mut new_descendants = BTreeSet::new();
        let ascendants_inds = ancestors
            .iter()
            .map(|&n| *topsort.get(n))
            .collect::<BTreeSet<_>>();
        for &descendant in &descendants {
            for port in graph.input_links(descendant).flatten() {
                let node = graph.port_node(port).expect("invalid port");
                match topsort.get(node) {
                    ind if ascendants_inds.contains(ind) => return true,
                    ind if ascendants_inds.first().unwrap() < ind => {
                        new_descendants.insert(node);
                    }
                    _ => (),
                };
            }
        }
        descendants = new_descendants;
    }
    false
}

#[cfg(test)]
mod tests {
    use portgraph::{algorithms::toposort, Direction, PortGraph, UnmanagedDenseMap};

    use super::is_ancestor;

    #[test]
    fn test_causal() {
        let mut g = PortGraph::new();
        let nodes = [(0, 2), (1, 2), (1, 2), (1, 0), (2, 0), (1, 0)]
            .into_iter()
            .map(|(i, o)| g.add_node(i, o))
            .collect::<Vec<_>>();
        g.link_nodes(nodes[0], 0, nodes[1], 0).unwrap();
        g.link_nodes(nodes[0], 1, nodes[2], 0).unwrap();
        g.link_nodes(nodes[1], 0, nodes[3], 0).unwrap();
        g.link_nodes(nodes[1], 1, nodes[4], 0).unwrap();
        g.link_nodes(nodes[2], 0, nodes[4], 1).unwrap();
        g.link_nodes(nodes[2], 1, nodes[5], 0).unwrap();
        let mut topsort = UnmanagedDenseMap::new();
        for (i, n) in
            toposort::<UnmanagedDenseMap<_, _>>(&g, [nodes[0]], Direction::Outgoing).enumerate()
        {
            topsort[n] = i;
        }
        for i in 1..=5 {
            assert!(is_ancestor(nodes[0], nodes[i], &g, &topsort));
            assert!(!is_ancestor(nodes[i], nodes[0], &g, &topsort));
        }
        assert!(is_ancestor(nodes[1], nodes[3], &g, &topsort));
        assert!(!is_ancestor(nodes[1], nodes[2], &g, &topsort));
        assert!(!is_ancestor(nodes[1], nodes[5], &g, &topsort));
        assert!(!is_ancestor(nodes[5], nodes[1], &g, &topsort));
        assert!(!is_ancestor(nodes[3], nodes[5], &g, &topsort));
    }
}
