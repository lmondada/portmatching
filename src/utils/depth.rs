use portgraph::{LinkView, NodeIndex, PortGraph, PortMut, PortView};

use super::pre_order::{Direction, PreOrder};

fn undirected_depths(graph: &PortGraph, start: NodeIndex) -> Vec<u32> {
    let preorder = PreOrder::new(graph, vec![start], Direction::Both);
    let mut depths = vec![u32::MAX; graph.node_capacity()];
    depths[start.index()] = 1;
    for node in preorder {
        let neighs = graph.all_neighbours(node);
        let min_depth = neighs
            .map(|neigh| depths[neigh.index()])
            .min()
            .unwrap_or(u32::MAX);
        if min_depth < u32::MAX {
            depths[node.index()] = min_depth + 1;
        }
    }
    depths
}

/// Returns true if the graph is connected.
pub fn is_connected(graph: &PortGraph) -> bool {
    let Some(root) = graph.nodes_iter().next() else {
        // An empty graph is connected
        return true;
    };
    undirected_depths(graph, root)
        .into_iter()
        .filter(|d| *d < u32::MAX)
        .count()
        == graph.node_count()
}

// TODO: the following might be useful in the near future
#[derive(Debug, PartialEq, Eq)]
pub enum NoCentreError {
    _DisconnectedGraph,
    _EmptyGraph,
}

pub fn _centre(graph: &PortGraph) -> Result<NodeIndex, NoCentreError> {
    let mut new2old = vec![NodeIndex::new(4); graph.node_capacity()];
    let rekey = |old: NodeIndex, new: NodeIndex| {
        new2old[new.index()] = old;
    };
    let mut graph = graph.clone();
    graph.compact_nodes(rekey);

    if graph.node_count() == 0 {
        return Err(NoCentreError::_EmptyGraph);
    } else if !is_connected(&graph) {
        return Err(NoCentreError::_DisconnectedGraph);
    }

    let centre = graph
        .nodes_iter()
        .min_by_key(|node| {
            let depths = undirected_depths(&graph, *node);
            depths.into_iter().filter(|d| *d < u32::MAX).max().unwrap()
        })
        .unwrap();
    Ok(new2old[centre.index()])
}

#[cfg(test)]
mod tests {
    use portgraph::{PortMut, PortView};

    use crate::utils::depth::{_centre, undirected_depths, NoCentreError};
    use crate::utils::test::*;

    #[test]
    fn depths() {
        let mut g = graph();
        g.compact_nodes(|_, _| {});
        g.shrink_to_fit();
        let v2 = g.nodes_iter().nth(2).unwrap();
        assert_eq!(undirected_depths(&g, v2), vec![2, 2, 1, 2]);
    }

    #[test]
    fn test_centre() {
        let g = graph();
        let v2 = g.nodes_iter().nth(2).unwrap();
        assert_eq!(_centre(&g).unwrap(), v2);
    }

    #[test]
    fn test_centre_disconnected() {
        let mut g = graph();
        g.add_node(2, 2);
        assert_eq!(_centre(&g).unwrap_err(), NoCentreError::_DisconnectedGraph);
    }
}
