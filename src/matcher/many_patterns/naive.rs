use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Debug},
};

use bimap::BiBTreeMap;
use portgraph::{portgraph::PortOffset, NodeIndex, PortGraph, PortIndex};

use super::{ManyPatternMatcher, ReadGraphTrie, WriteGraphTrie};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternNodeAddress(usize, usize);

impl PatternNodeAddress {
    const ROOT: PatternNodeAddress = Self(0, 0);

    fn next(&self) -> Self {
        Self(self.0, self.1 + 1)
    }

    fn next_root(&self) -> Self {
        Self(self.0 + 1, self.1)
    }
}

impl Debug for PatternNodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", (self.0, self.1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TreeNodeID {
    line_tree: usize,
    ind: usize,
}

impl TreeNodeID {
    fn new(line_tree: usize, ind: usize) -> Self {
        Self { line_tree, ind }
    }

    fn line_root(line_tree: usize) -> Self {
        Self { line_tree, ind: 0 }
    }

    const ROOT: TreeNodeID = TreeNodeID {
        line_tree: 0,
        ind: 0,
    };
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum NodeTransition {
    KnownNode(PatternNodeAddress, PortOffset),
    NewNode(PortOffset),
    NoLinkedNode,
    Fail,
}

impl fmt::Display for NodeTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeTransition::KnownNode(PatternNodeAddress(i, j), port) => {
                write!(f, "({}, {})[{:?}]", i, j, port)
            }
            NodeTransition::NewNode(port) => {
                write!(f, "X[{:?}]", port)
            }
            NodeTransition::NoLinkedNode => write!(f, "dangling"),
            NodeTransition::Fail => write!(f, "fail"),
        }
    }
}

#[derive(Clone, Default)]
struct TreeNode {
    port_offset: Option<PortOffset>,
    address: Option<PatternNodeAddress>,
    transitions: BTreeMap<NodeTransition, TreeNodeID>,
}

pub struct NaiveGraphTrie(Vec<Vec<TreeNode>>);

impl Default for NaiveGraphTrie {
    fn default() -> Self {
        Self(vec![vec![TreeNode::root()]])
    }
}

impl NaiveGraphTrie {
    pub fn new() -> Self {
        Self::default()
    }

    fn transition(&self, node: &TreeNodeID, transition: &NodeTransition) -> Option<&TreeNodeID> {
        self.state(node).transitions.get(transition)
    }

    fn set_transition(
        &mut self,
        node: &TreeNodeID,
        next_node: &TreeNodeID,
        transition: &NodeTransition,
    ) -> &TreeNodeID {
        let old = self
            .state_mut(node)
            .transitions
            .insert(transition.clone(), next_node.clone());
        if old.is_some() && old != Some(next_node.clone()) {
            panic!("Changing transition value");
        }
        self.transition(node, &transition).unwrap()
    }

    fn state(&self, node: &TreeNodeID) -> &TreeNode {
        &self.0[node.line_tree][node.ind]
    }

    fn state_mut(&mut self, node: &TreeNodeID) -> &mut TreeNode {
        &mut self.0[node.line_tree][node.ind]
    }

    pub fn clone_into(&mut self, node: &TreeNodeID, node_to: &TreeNodeID) {
        self.state_mut(node_to).port_offset = self.state(node).port_offset.clone();
        self.state_mut(node_to).address = self.state(node).address.clone();
    }

    fn append_tree(&mut self, tree_ind: usize) -> TreeNodeID {
        let ind = self.0[tree_ind].len();
        self.0[tree_ind].push(TreeNode::new());
        TreeNodeID::new(tree_ind, ind)
    }

    fn add_new_tree(&mut self) -> TreeNodeID {
        let node_id = TreeNodeID::line_root(self.0.len());
        self.0.push(vec![TreeNode::new()]);
        node_id
    }

    fn extend_tree(&mut self, node: &TreeNodeID) -> TreeNodeID {
        let next_node = self.append_tree(node.line_tree);
        let fallback = self
            .transition(node, &NodeTransition::NoLinkedNode)
            .or(self.transition(node, &NodeTransition::Fail))
            .cloned();
        if let Some(fallback) = fallback {
            let subtree = self.clone_tree(&fallback);
            self.set_transition(&next_node, &subtree, &NodeTransition::NoLinkedNode);
            let subtree = self.clone_tree(&fallback);
            self.set_transition(&next_node, &subtree, &NodeTransition::Fail);
        }
        next_node
    }

    fn is_root(root: &TreeNodeID) -> bool {
        root.ind == 0
    }

    fn clone_tree(&mut self, old_root: &TreeNodeID) -> TreeNodeID {
        assert!(Self::is_root(old_root));
        let mut line_trees = VecDeque::from([old_root.line_tree]);
        let new_root = TreeNodeID::new(self.0.len(), 0);

        while let Some(old_line_tree) = line_trees.pop_front() {
            let curr_line_tree = self.0.len();
            // Clone tree
            self.0.push(self.0[old_line_tree].clone());
            // Update transitions
            for node in self.0[curr_line_tree].iter_mut() {
                for (label, next_node) in node.transitions.iter_mut() {
                    match label {
                        NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                            next_node.line_tree = curr_line_tree;
                        }
                        NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                            line_trees.push_back(next_node.line_tree);
                            next_node.line_tree = curr_line_tree + line_trees.len();
                        }
                    }
                }
            }
        }
        new_root
    }

    pub fn dotstring(&self) -> String {
        let mut nodes = String::new();
        let mut edges = String::new();
        let mut ranks = String::new();
        let to_str = |i, j| format!("A{}at{}", i, j);
        for (i, line_tree) in self.0.iter().enumerate() {
            for (j, node) in line_tree.iter().enumerate() {
                nodes += &to_str(i, j);
                if let Some(address) = &node.address {
                    nodes += &format!(" [label=\"({}, {})", address.0, address.1);
                } else {
                    nodes += " [label=\"None";
                }
                if let Some(out_port) = &node.port_offset {
                    nodes += &format!("[{:?}]", out_port,);
                }
                nodes += "\"];\n";
                for (label, target) in node.transitions.iter() {
                    let (new_i, new_j) = (target.line_tree, target.ind);
                    edges += &format!(
                        "A{}at{} -> A{}at{} [label=\"{}\"];\n",
                        i, j, new_i, new_j, label
                    );
                }
            }
            ranks += &format!(
                "{{ rank=same; {} }}\n",
                &(0..line_tree.len())
                    .map(|j| to_str(i, j))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let mut ret = "digraph pattern_trie {\n".to_string();
        ret += &nodes;
        ret += &edges;
        ret += &ranks;
        ret += "}";
        ret
    }

    fn compute_transition_for_state(
        &self,
        state: &TreeNodeID,
        graph: &PortGraph,
        mapped_nodes: &BiBTreeMap<NodeIndex, PatternNodeAddress>,
    ) -> Option<(NodeTransition, Option<NodeIndex>)> {
        let addr = self.state(state).address.as_ref()?;
        let graph_node = *mapped_nodes
            .get_by_right(addr)
            .expect("Malformed pattern trie");
        let out_port = self.state(state).port_offset.clone()?;
        let out_port = graph.port_alt(graph_node, out_port)?;
        let in_port = graph.port_link(out_port);
        (
            compute_transition(in_port, graph, |n| mapped_nodes.get_by_left(n).cloned()),
            in_port.and_then(|in_port| graph.port_node(in_port)),
        )
            .into()
    }

    fn try_update_node(
        &mut self,
        node: &TreeNodeID,
        addr: &PatternNodeAddress,
        port_index: &PortOffset,
    ) -> bool {
        let node = self.state_mut(node);
        let res_port_index = node.port_offset.get_or_insert(port_index.clone());
        let res_addr = node.address.get_or_insert(addr.clone());
        res_port_index == port_index && res_addr == addr
    }

    fn find_root(
        &mut self,
        root: &TreeNodeID,
        addr: &PatternNodeAddress,
        port_index: &PortOffset,
    ) -> TreeNodeID {
        let mut root = root.clone();
        while !self.try_update_node(&root, addr, port_index) {
            root = match self.transition(&root, &NodeTransition::Fail).cloned() {
                Some(next_tree) => next_tree,
                None => {
                    let next_tree = self.add_new_tree();
                    self.set_transition(&root, &next_tree, &NodeTransition::Fail);
                    next_tree
                }
            };
            assert!(Self::is_root(&root));
        }
        root
    }
}

fn compute_transition<F: Fn(&NodeIndex) -> Option<PatternNodeAddress>>(
    in_port: Option<PortIndex>,
    graph: &PortGraph,
    get_addr: F,
) -> NodeTransition {
    match in_port {
        Some(in_port) => {
            let port_offset = graph.port_index(in_port).unwrap();
            let in_node = graph.port_node(in_port).unwrap();
            match get_addr(&in_node) {
                Some(new_addr) => NodeTransition::KnownNode(new_addr.clone(), port_offset),
                None => NodeTransition::NewNode(port_offset),
            }
        }
        None => NodeTransition::NoLinkedNode,
    }
}

impl TreeNode {
    fn new() -> TreeNode {
        Self::default()
    }

    fn with_address(addr: PatternNodeAddress) -> Self {
        Self {
            port_offset: None,
            address: addr.into(),
            transitions: [].into(),
        }
    }

    fn root() -> TreeNode {
        Self::with_address(PatternNodeAddress(0, 0))
    }
}

#[derive(Clone)]
pub struct MatchObject {
    map: BiBTreeMap<NodeIndex, PatternNodeAddress>,
    // Entries in `no_map` are in `map` but are irrelevant
    no_map: BTreeSet<PatternNodeAddress>,
    current_addr: PatternNodeAddress,
}

impl MatchObject {
    fn new(left_root: NodeIndex) -> Self {
        Self {
            map: BiBTreeMap::from_iter([(left_root, PatternNodeAddress::ROOT)]),
            no_map: BTreeSet::new(),
            current_addr: PatternNodeAddress::ROOT,
        }
    }
}

fn valid_write_transitions(
    in_port: PortIndex,
    graph: &PortGraph,
    current_match: &MatchObject,
) -> Vec<(NodeTransition, MatchObject)> {
    let ideal_transition = compute_transition(Some(in_port), graph, |n| {
        current_match.map.get_by_left(n).cloned()
    });

    // Compute the address of the next state
    let next_addr = current_match.current_addr.next();

    let mut transitions = Vec::new();
    if let NodeTransition::NewNode(port) = &ideal_transition {
        transitions.extend(
            current_match
                .no_map
                .iter()
                .map(|addr| NodeTransition::KnownNode(addr.clone(), port.clone())),
        );
    }
    transitions.push(ideal_transition);
    transitions
        .into_iter()
        .map(|transition| {
            let mut next_match = current_match.clone();
            next_match.current_addr = next_addr.clone();
            if let NodeTransition::NewNode(_) = &transition {
                let next_graph_node = graph.port_node(in_port).expect("Invalid port");
                next_match.map.insert(next_graph_node, next_addr.clone());
            }
            (transition, next_match)
        })
        .collect()
}

impl WriteGraphTrie for NaiveGraphTrie {
    fn create_next_states(
        &mut self,
        state: &Self::StateID,
        edge: (PortIndex, PortIndex),
        graph: &PortGraph,
        current_match: &Self::MatchObject,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        // We start by finding where we are in the graph, using the outgoing
        // port edge.0
        let graph_node = graph.port_node(edge.0).expect("Invalid port");
        let graph_addr = current_match
            .map
            .get_by_left(&graph_node)
            .expect("Incomplete match map");
        let graph_port = graph.port_index(edge.0).expect("Invalid port");

        // If we are at the root of a tree, we now find the state that
        // corresponds to our position in the graph
        let mut state = state.clone();
        if Self::is_root(&state) {
            state = self.find_root(&state, graph_addr, &graph_port);
        }

        // Check that the address in the graph and the address in the trie
        // are identical.
        // If the state has no address, then we can set it to the one we need
        let state_addr = self
            .state_mut(&state)
            .address
            .get_or_insert(graph_addr.clone());
        assert_eq!(graph_addr, state_addr);

        // Same for the port offset
        let state_port = self
            .state_mut(&state)
            .port_offset
            .get_or_insert(graph_port.clone());
        assert_eq!(&graph_port, state_port);

        // For every allowable transition, insert it if it does not exist
        // and return it
        valid_write_transitions(edge.1, graph, current_match)
            .into_iter()
            .map(|(transition, next_match)| {
                (
                    self.transition(&state, &transition)
                        .cloned()
                        .unwrap_or_else(|| {
                            let next_state = self.extend_tree(&state);
                            self.state_mut(&next_state).address =
                                next_match.current_addr.clone().into();
                            self.set_transition(&state, &next_state, &transition);
                            next_state
                        }),
                    next_match,
                )
            })
            .collect()
    }

    fn create_next_roots(
        &mut self,
        state: &Self::StateID,
        current_match: &Self::MatchObject,
        is_dangling: bool,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        let mut current_states = vec![(state.clone(), current_match.clone())];
        let mut next_states = Vec::new();
        assert!(!Self::is_root(state));

        while let Some((current_state, current_match)) = current_states.pop() {
            for (transition, next_state) in &self.state(&current_state).transitions {
                match transition {
                    NodeTransition::KnownNode(_, _) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        current_states.push((next_state.clone(), next_match));
                    }
                    NodeTransition::NewNode(_) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr.clone();
                        next_match.no_map.insert(next_addr);
                        current_states.push((next_state.clone(), next_match));
                    }
                    NodeTransition::NoLinkedNode => {
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((next_state.clone(), next_match));
                    }
                    NodeTransition::Fail => {
                        if is_dangling && &current_state == state {
                            continue;
                        }
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((next_state.clone(), next_match));
                    }
                }
            }
        }
        next_states
    }
}

impl ReadGraphTrie for NaiveGraphTrie {
    type StateID = TreeNodeID;

    type MatchObject = MatchObject;

    fn init(&self, root: NodeIndex) -> (Self::StateID, Self::MatchObject) {
        (TreeNodeID::ROOT, MatchObject::new(root))
    }

    fn next_states(
        &self,
        state: &Self::StateID,
        graph: &PortGraph,
        current_match: &Self::MatchObject,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        let mut next_states = Vec::new();
        let mut state = Some(state.clone());
        while let Some(curr_state) = state {
            // Compute "ideal" transition
            let (mut transition, mut next_node) = self
                .compute_transition_for_state(&curr_state, graph, &current_match.map)
                // Default to Fail transition
                .unwrap_or((NodeTransition::Fail, None));
            // Fall-back to simpler transitions if non-existent
            while self.transition(&curr_state, &transition).is_none()
                && transition != NodeTransition::Fail
            {
                transition = match transition {
                    NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                        NodeTransition::NoLinkedNode
                    }
                    NodeTransition::NoLinkedNode => NodeTransition::Fail,
                    NodeTransition::Fail => unreachable!(),
                };
                next_node = None;
            }
            // Add transition to next_states
            if let Some(next_state) = self.transition(&curr_state, &transition) {
                let mut next_match = current_match.clone();
                match (next_node, &self.state(&next_state).address) {
                    (Some(next_node), Some(next_addr)) => {
                        dbg!(&next_match.map);
                        println!("inserting ({:?} <> {:?})", next_node, next_addr);
                        next_match
                            .map
                            .insert_no_overwrite(next_node, next_addr.clone())
                            .ok()
                            .or_else(|| {
                                (next_match.map.get_by_left(&next_node) == Some(next_addr))
                                    .then_some(())
                            })
                            .expect("Map is not injective");
                    }
                    _ => {}
                }
                next_states.push((next_state.clone(), next_match));
            }
            // Repeat if we are at a root (i.e. non-deterministic)
            if curr_state.ind == 0 {
                state = self.transition(&curr_state, &NodeTransition::Fail).cloned();
            } else {
                break;
            }
        }
        next_states
    }
}

pub type NaiveManyPatternMatcher = ManyPatternMatcher<NaiveGraphTrie>;

#[cfg(test)]
mod tests {
    use std::fs;

    use itertools::Itertools;
    use portgraph::{Direction, NodeIndex, PortGraph};

    use proptest::prelude::*;

    use crate::{
        matcher::{
            many_patterns::{
                naive::{NaiveManyPatternMatcher},
                PatternID, PatternMatch,
            },
            Matcher, SinglePatternMatcher,
        },
        pattern::Pattern,
        utils::test_utils::{arb_portgraph_connected, non_empty_portgraph},
    };

    #[test]
    fn a_few_patterns() {
        let mut trie = NaiveManyPatternMatcher::new();

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(1, 1);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        let pattern = Pattern::from_graph(g.clone()).unwrap();
        trie.add_pattern(pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(1, 1);
        let v2 = g.add_node(1, 1);
        let v3 = g.add_node(1, 0);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        let v1_out0 = g.port(v1, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in0 = g.port(v2, 0, portgraph::Direction::Incoming).unwrap();
        let v2_out0 = g.port(v2, 0, portgraph::Direction::Outgoing).unwrap();
        let v3_in0 = g.port(v3, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        g.link_ports(v1_out0, v2_in0).unwrap();
        g.link_ports(v2_out0, v3_in0).unwrap();
        let mut pattern = Pattern::from_graph(g.clone()).unwrap();
        pattern.root = v0;
        trie.add_pattern(pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(2, 0);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        let pattern = Pattern::from_graph(g.clone()).unwrap();
        trie.add_pattern(pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(2, 1);
        let v2 = g.add_node(1, 0);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        let v1_out0 = g.port(v1, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in0 = g.port(v2, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        g.link_ports(v1_out0, v2_in0).unwrap();
        let mut pattern = Pattern::from_graph(g.clone()).unwrap();
        pattern.root = v0;
        trie.add_pattern(pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(1, 1);
        let v2 = g.add_node(2, 2);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        let v1_out0 = g.port(v1, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in1 = g.port(v2, 1, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        g.link_ports(v1_out0, v2_in1).unwrap();
        let mut pattern = Pattern::from_graph(g.clone()).unwrap();
        pattern.root = v0;
        trie.add_pattern(pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(1, 1);
        let v2 = g.add_node(2, 2);
        let v3 = g.add_node(0, 1);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        let v1_out0 = g.port(v1, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in1 = g.port(v2, 1, portgraph::Direction::Incoming).unwrap();
        let v3_out0 = g.port(v3, 0, portgraph::Direction::Outgoing).unwrap();
        let v2_in0 = g.port(v2, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        g.link_ports(v1_out0, v2_in1).unwrap();
        g.link_ports(v3_out0, v2_in0).unwrap();
        let mut pattern = Pattern::from_graph(g.clone()).unwrap();
        pattern.root = v0;
        trie.add_pattern(pattern);

        let expected_trie = fs::read_to_string("patterntrie_baby.gv").unwrap();
        assert_eq!(expected_trie, trie.trie.dotstring());
    }

    #[test]
    fn single_pattern_loop_link() {
        let mut g = PortGraph::new();
        g.add_node(0, 1);
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port(n, 0, Direction::Outgoing).unwrap(),
            g.port(n, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                id: PatternID(0),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_loop_link2() {
        let mut g = PortGraph::new();
        let n = g.add_node(1, 1);
        g.link_ports(
            g.port(n, 0, Direction::Outgoing).unwrap(),
            g.port(n, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();
        let p = Pattern::from_graph(g.clone()).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);

        assert_eq!(
            matcher.find_matches(&g),
            vec![PatternMatch {
                id: PatternID(0),
                root: NodeIndex::new(0)
            }]
        );
    }

    #[test]
    fn single_pattern_simple() {
        let mut g = PortGraph::new();
        g.add_node(0, 2);
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);

        let mut g = PortGraph::new();
        let n0 = g.add_node(0, 1);
        let n1 = g.add_node(1, 0);
        g.link_ports(
            g.port(n0, 0, Direction::Outgoing).unwrap(),
            g.port(n1, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    fn link(p: &mut PortGraph, (n1, p1): (NodeIndex, usize), (n2, p2): (NodeIndex, usize)) {
        p.link_ports(
            p.port(n1, p1, Direction::Outgoing).unwrap(),
            p.port(n2, p2, Direction::Incoming).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn two_simple_patterns() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(0, 1);
        let n1 = p1.add_node(2, 0);
        let n2 = p1.add_node(0, 1);
        link(&mut p1, (n0, 0), (n1, 0));
        link(&mut p1, (n2, 0), (n1, 1));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n0 = g.add_node(2, 2);
        let n1 = g.add_node(3, 1);
        let n2 = g.add_node(0, 2);
        let n3 = g.add_node(3, 2);
        link(&mut g, (n0, 0), (n1, 2));
        link(&mut g, (n0, 1), (n3, 0));
        link(&mut g, (n1, 0), (n3, 1));
        link(&mut g, (n2, 0), (n1, 0));
        link(&mut g, (n2, 1), (n3, 2));
        link(&mut g, (n3, 0), (n1, 1));
        link(&mut g, (n3, 1), (n0, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
        fs::write("graph2.gv", g.dotstring()).unwrap();
        fs::write("patterntrie.gv", matcher.trie.dotstring()).unwrap();
        assert_eq!(matcher.find_matches(&g).len(), 3);
    }

    #[test]
    fn two_simple_patterns_2() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(1, 0);
        let n1 = p1.add_node(1, 3);
        let n2 = p1.add_node(0, 3);
        let n3 = p1.add_node(1, 3);
        link(&mut p1, (n1, 0), (n0, 0));
        link(&mut p1, (n2, 0), (n1, 0));
        link(&mut p1, (n2, 2), (n3, 0));

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
        assert_eq!(matcher.find_matches(&g).len(), 1);
    }

    proptest! {
        #[ignore = "no proptests"]
        #[test]
        fn single_graph_proptest(pattern in arb_portgraph_connected(10, 4, 20), g in non_empty_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_graph(pattern).unwrap();
            let mut matcher = NaiveManyPatternMatcher::new();
            let pattern_id = matcher.add_pattern(pattern.clone());
            let single_matcher = SinglePatternMatcher::from_pattern(pattern.clone());
            let many_matches = matcher.find_matches(&g);
            let single_matches: Vec<_> = single_matcher
                .find_matches(&g)
                .into_iter()
                .map(|m| PatternMatch {
                    id: pattern_id,
                    root: m[&pattern.root],
                })
                .collect();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    proptest! {
        #[ignore = "no proptests"]
        #[test]
        fn many_graphs_proptest(
            patterns in prop::collection::vec(arb_portgraph_connected(10, 4, 20), 1..10),
            g in non_empty_portgraph(100, 4, 200)
        ) {
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            let matcher = NaiveManyPatternMatcher::from_patterns(patterns.clone());
            let single_matchers = patterns
                .clone()
                .into_iter()
                .map(SinglePatternMatcher::from_pattern)
                .collect_vec();
            let many_matches = matcher.find_matches(&g);
            let many_matches = (0..patterns.len())
                .map(|i| {
                    many_matches
                        .iter()
                        .filter(|m| m.id == PatternID(i))
                        .cloned()
                        .collect_vec()
                })
                .collect_vec();
            let single_matches = single_matchers
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    m.find_matches(&g)
                        .into_iter()
                        .map(|m| PatternMatch {
                            id: PatternID(i),
                            root: m[&patterns[i].root],
                        })
                        .collect_vec()
                })
                .collect_vec();
            for (i, p) in patterns.iter().enumerate() {
                fs::write(format!("p{}.gv", i), p.graph.dotstring()).unwrap();
            }
            fs::write("graph.gv", g.dotstring()).unwrap();
            fs::write("patterntrie.gv", matcher.trie.dotstring()).unwrap();
            prop_assert_eq!(many_matches, single_matches);
        }
    }

    // #[test]
    // fn traverse_from_test() {
    //     let mut matcher = ManyPatternMatcher::new();
    //     let tree = &mut matcher.tree.line_trees[0];
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(0)),
    //         TreeNodeID::SameTree(1),
    //     );
    //     tree[0].transitions.insert(
    //         NodeTransition::NewNode(PortOffset(1)),
    //         TreeNodeID::SameTree(2),
    //     );
    //     tree.push(TreeNode {
    //         out_port: Some(PortOffset(1)),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(
    //             NodeTransition::NewNode(PortOffset(0)),
    //             TreeNodeID::SameTree(3),
    //         )]
    //         .into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(2).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(1).into(),
    //         address: PatternNodeAddress(0, 2),
    //         transitions: [].into(),
    //         matches: vec![],
    //     });
    //     tree.push(TreeNode {
    //         out_port: PortOffset(0).into(),
    //         address: PatternNodeAddress(0, 1),
    //         transitions: [(NodeTransition::NoLinkedNode, TreeNodeID::SameTree(4))].into(),
    //         matches: vec![],
    //     });

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v3 = g.add_node(1, 1);
    //     let v4 = g.add_node(1, 0);
    //     let v0_0 = g.port(v0, 0, Direction::Outgoing).unwrap();
    //     let v1_0 = g.port(v1, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v0_0, v1_0).unwrap();
    //     let v1_1 = g.port(v1, 0, Direction::Outgoing).unwrap();
    //     let v3_0 = g.port(v3, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v1_1, v3_0).unwrap();
    //     let v3_1 = g.port(v3, 0, Direction::Outgoing).unwrap();
    //     let v4_0 = g.port(v4, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v3_1, v4_0).unwrap();
    //     let line = vec![
    //         Edge(v0_0, v1_0.into()),
    //         Edge(v1_1, v3_0.into()),
    //         Edge(v3_1, v4_0.into()),
    //     ];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line),
    //         (3, Some(&Edge(v3_1, v4_0.into())))
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v1, PatternNodeAddress(0, 1)),
    //             (v3, PatternNodeAddress(0, 2)),
    //         ]
    //         .into()
    //     );

    //     let mut g = PortGraph::new();
    //     let v0 = g.add_node(0, 2);
    //     let v1 = g.add_node(1, 1);
    //     let v2 = g.add_node(2, 1);
    //     let v0_0 = g.port(v0, 0, Direction::Outgoing).unwrap();
    //     let v2_1 = g.port(v2, 1, Direction::Incoming).unwrap();
    //     let v2_2 = g.port(v2, 0, Direction::Outgoing).unwrap();
    //     g.link_ports(v0_0, v2_1).unwrap();
    //     let line = vec![Edge(v0_0, v2_1.into()), Edge(v2_2, None)];
    //     let pattern = Pattern { graph: g, root: v0 };
    //     let mut transitioner = TransitionCalculator::new(&pattern);
    //     assert_eq!(
    //         transitioner.traverse_line(tree, &line, TransitionPolicy::NoDanglingEdge),
    //         (4, None)
    //     );
    //     assert_eq!(
    //         transitioner.mapped_nodes,
    //         [
    //             (v0, PatternNodeAddress(0, 0)),
    //             (v2, PatternNodeAddress(0, 1)),
    //         ]
    //         .into()
    //     );
    // }
}
