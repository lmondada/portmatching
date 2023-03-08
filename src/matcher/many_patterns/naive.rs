use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Debug},
};

use bimap::BiBTreeMap;
use portgraph::{portgraph::PortOffset, NodeIndex, PortGraph, PortIndex};

use crate::pattern::{Edge, Pattern};

use crate::matcher::Matcher;

use super::{PatternID, PatternMatch, StateAutomatonMatcher};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternNodeAddress(usize, usize);

impl Debug for PatternNodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", (self.0, self.1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    fn is_same_tree(&self, other: &TreeNodeID) -> bool {
        self.line_tree == other.line_tree
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
    out_port: Option<PortOffset>,
    address: Option<PatternNodeAddress>,
    transitions: BTreeMap<NodeTransition, TreeNodeID>,
    matches: Vec<PatternID>,
    // The root is non-deterministic, everything else is deterministic
    // deterministic: bool,
}

struct PatternTrie {
    line_trees: Vec<Vec<TreeNode>>,
    next_pattern_id: usize,
}

impl PatternTrie {
    fn new() -> Self {
        Self {
            line_trees: vec![vec![TreeNode::root()]],
            next_pattern_id: 0,
        }
    }

    pub fn transition(
        &self,
        node: &TreeNodeID,
        transition: &NodeTransition,
    ) -> Option<&TreeNodeID> {
        self.node(node).transitions.get(transition)
    }

    pub fn set_transition(
        &mut self,
        node: &TreeNodeID,
        next_node: &TreeNodeID,
        transition: &NodeTransition,
    ) -> &TreeNodeID {
        let old = self
            .node_mut(node)
            .transitions
            .insert(transition.clone(), next_node.clone());
        if old.is_some() && old != Some(next_node.clone()) {
            panic!("Changing transition value");
        }
        self.transition(node, &transition).unwrap()
    }

    pub fn node(&self, node: &TreeNodeID) -> &TreeNode {
        &self.line_trees[node.line_tree][node.ind]
    }

    pub fn node_mut(&mut self, node: &TreeNodeID) -> &mut TreeNode {
        &mut self.line_trees[node.line_tree][node.ind]
    }

    pub fn clone_into(&mut self, node: &TreeNodeID, node_to: &TreeNodeID) {
        self.node_mut(node_to).out_port = self.node(node).out_port.clone();
        self.node_mut(node_to).address = self.node(node).address.clone();
        self.node_mut(node_to).matches = self.node(node).matches.clone();
    }

    fn append_tree(&mut self, tree_ind: usize) -> TreeNodeID {
        let ind = self.line_trees[tree_ind].len();
        self.line_trees[tree_ind].push(TreeNode::new());
        TreeNodeID::new(tree_ind, ind)
    }

    fn add_new_tree(&mut self) -> TreeNodeID {
        let node_id = TreeNodeID::line_root(self.line_trees.len());
        self.line_trees.push(vec![TreeNode::new()]);
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

    fn all_descendents_in_tree_with_ref(&self, root: &TreeNodeID, nodes: &mut Vec<TreeNodeID>) {
        nodes.push(root.clone());
        for child in self.node(root).transitions.values() {
            if child.is_same_tree(root) {
                self.all_descendents_in_tree_with_ref(child, nodes)
            }
        }
    }

    fn all_descendents_in_tree(&self, root: &TreeNodeID) -> Vec<TreeNodeID> {
        let mut nodes = Vec::new();
        self.all_descendents_in_tree_with_ref(root, &mut nodes);
        nodes
    }

    pub fn is_root(root: &TreeNodeID) -> bool {
        root.ind == 0
    }

    fn clone_tree(&mut self, old_root: &TreeNodeID) -> TreeNodeID {
        assert!(Self::is_root(old_root));
        let mut line_trees = VecDeque::from([old_root.line_tree]);
        let new_root = TreeNodeID::new(self.line_trees.len(), 0);

        while let Some(old_line_tree) = line_trees.pop_front() {
            let curr_line_tree = self.line_trees.len();
            // Clone tree
            self.line_trees.push(self.line_trees[old_line_tree].clone());
            // Update transitions
            for node in self.line_trees[curr_line_tree].iter_mut() {
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

    pub fn add_pattern(&mut self, pattern: &Pattern) -> Option<PatternID> {
        let graph = &pattern.graph;
        let mut mapped_nodes = BTreeMap::new();
        let all_lines = pattern.all_lines();
        let mut next_trees = Vec::from([TreeNodeID::ROOT]);
        for (line_ind, line) in all_lines.into_iter().enumerate() {
            let mut new_next_trees = Vec::new();
            let mut new_mapped_nodes = mapped_nodes.clone();
            for next_tree in next_trees {
                new_mapped_nodes = mapped_nodes.clone();
                new_next_trees.append(&mut self.add_line(
                    next_tree,
                    &line,
                    line_ind,
                    graph,
                    &mut new_mapped_nodes,
                    &BTreeSet::new()
                ));
            }
            mapped_nodes = new_mapped_nodes;
            next_trees = new_next_trees;
        }
        let pattern_id = PatternID(self.next_pattern_id);
        self.next_pattern_id += 1;

        for next_tree in next_trees {
            self.node_mut(&next_tree).matches.push(pattern_id);
        }
        pattern_id.into()
    }

    fn add_line(
        &mut self,
        mut root: TreeNodeID,
        line: &[Edge],
        line_ind: usize,
        graph: &PortGraph,
        mapped_nodes: &mut BTreeMap<NodeIndex, PatternNodeAddress>,
        non_mapped_addresses: &BTreeSet<PatternNodeAddress>,
    ) -> Vec<TreeNodeID> {
        let Some(first_port) = line.first().map(|e| e.0) else {
            return vec![];
        };

        // Find root with the correct port_index and address
        let first_port_index = graph.port_index(first_port).expect("Invalid edge line");
        let first_node = graph.port_node(first_port).expect("Invalid edge line");
        let default_addr = PatternNodeAddress(line_ind, 0);
        let first_addr =
            mapped_nodes.entry(first_node).or_insert(default_addr) as &PatternNodeAddress;
        self.find_root(&mut root, first_addr, &first_port_index);

        // Traverse line
        let mut curr_nodes = vec![root];
        let mut last_is_dangling = false;
        for (ind, &Edge(out_port, in_port)) in line.into_iter().enumerate() {
            let Some(in_port) = in_port else {
                // We've reached the last edge of the line -- a dangling edge
                last_is_dangling = true;
                break;
            };
            let mut next_nodes = Vec::new();
            for curr_node in curr_nodes {
                let next_addr = PatternNodeAddress(line_ind, ind + 1);
                next_nodes.append(&mut self.traverse_edge(
                    curr_node,
                    out_port,
                    in_port,
                    next_addr,
                    graph,
                    mapped_nodes,
                    non_mapped_addresses,
                ));
            }
            curr_nodes = next_nodes;
        }

        // Get descendants
        let mut all_descendents = Vec::new();
        for node in curr_nodes {
            all_descendents.append(&mut self.descendents(&node, last_is_dangling));
        }
        all_descendents
    }

    fn traverse_edge(
        &mut self,
        node: TreeNodeID,
        out_port: PortIndex,
        in_port: PortIndex,
        next_addr: PatternNodeAddress,
        graph: &PortGraph,
        mapped_nodes: &mut BTreeMap<NodeIndex, PatternNodeAddress>,
        non_mapped_addresses: &BTreeSet<PatternNodeAddress>,
    ) -> Vec<TreeNodeID> {
        // Check address is valid
        let graph_node = graph.port_node(out_port).expect("Invalid port");
        let curr_addr = self
            .node_mut(&node)
            .address
            .get_or_insert(mapped_nodes[&graph_node].clone());
        assert_eq!(&mapped_nodes[&graph_node], curr_addr);

        // Check out_port is valid
        let graph_port_index = graph.port_index(out_port).expect("Invalid port");
        let curr_port = self
            .node_mut(&node)
            .out_port
            .get_or_insert(graph_port_index.clone());
        assert_eq!(curr_port, &graph_port_index);

        match &compute_transition(Some(in_port), graph, |n| mapped_nodes.get(n).cloned()) {
            transition @ NodeTransition::KnownNode(addr, _) => vec![self
                .transition(&node, transition)
                .cloned()
                .unwrap_or_else(|| {
                    let next_node = self.extend_tree(&node);
                    self.node_mut(&next_node).address = addr.clone().into();
                    self.set_transition(&node, &next_node, &transition);
                    next_node
                })],
            transition @ NodeTransition::NewNode(port) => {
                let next_graph_node = graph.port_node(in_port).expect("Invalid port");
                mapped_nodes.insert(next_graph_node, next_addr.clone());
                let mut transitions = Vec::with_capacity(non_mapped_addresses.len() + 1);
                transitions.push(transition.clone());
                transitions.extend(
                    non_mapped_addresses
                        .iter()
                        .map(|addr| NodeTransition::KnownNode(addr.clone(), port.clone())),
                );
                transitions
                    .iter()
                    .map(|transition| {
                        self.transition(&node, &transition)
                            .cloned()
                            .unwrap_or_else(|| {
                                let next_node = self.extend_tree(&node);
                                self.node_mut(&next_node).address = next_addr.clone().into();
                                self.set_transition(&node, &next_node, transition);
                                next_node
                            })
                    })
                    .collect()
            }
            NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                panic!("compute_transition returned unexpected transition")
            }
        }
    }

    fn descendents(&mut self, root: &TreeNodeID, is_dangling: bool) -> Vec<TreeNodeID> {
        let mut next_trees = Vec::new();
        for node in self.all_descendents_in_tree(root) {
            for transition in [NodeTransition::Fail, NodeTransition::NoLinkedNode] {
                if transition == NodeTransition::Fail && &node == root && is_dangling {
                    continue;
                }
                let next_node = self
                    .transition(&node, &transition)
                    .cloned()
                    .unwrap_or_else(|| {
                        let next_node = self.add_new_tree();
                        self.set_transition(&node, &next_node, &transition);
                        next_node
                    });
                assert!(Self::is_root(&next_node));
                next_trees.push(next_node);
            }
        }
        next_trees
    }

    pub fn dotstring(&self) -> String {
        let mut nodes = String::new();
        let mut edges = String::new();
        let mut ranks = String::new();
        let to_str = |i, j| format!("A{}at{}", i, j);
        for (i, line_tree) in self.line_trees.iter().enumerate() {
            for (j, node) in line_tree.iter().enumerate() {
                nodes += &to_str(i, j);
                if let Some(address) = &node.address {
                    nodes += &format!(" [label=\"({}, {})", address.0, address.1);
                } else {
                    nodes += " [label=\"None";
                }
                if let Some(out_port) = &node.out_port {
                    nodes += &format!("[{:?}]", out_port,);
                }
                let matches = &self.node(&TreeNodeID::new(i, j)).matches;
                if !matches.is_empty() {
                    nodes += &format!("\n{:?}\"];\n", self.node(&TreeNodeID::new(i, j)).matches);
                } else {
                    nodes += "\"];\n";
                }
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
        let addr = self.node(state).address.as_ref()?;
        let graph_node = *mapped_nodes
            .get_by_right(addr)
            .expect("Malformed pattern trie");
        let out_port = self.node(state).out_port.clone()?;
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
        let node = self.node_mut(node);
        let res_port_index = node.out_port.get_or_insert(port_index.clone());
        let res_addr = node.address.get_or_insert(addr.clone());
        res_port_index == port_index && res_addr == addr
    }

    fn find_root(
        &mut self,
        root: &mut TreeNodeID,
        addr: &PatternNodeAddress,
        port_index: &PortOffset,
    ) {
        while !self.try_update_node(root, addr, port_index) {
            *root = match self.transition(&root, &NodeTransition::Fail).cloned() {
                Some(next_tree) => next_tree,
                None => {
                    let next_tree = self.add_new_tree();
                    self.set_transition(&root, &next_tree, &NodeTransition::Fail);
                    next_tree
                }
            };
            assert!(Self::is_root(&root));
        }
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
            out_port: None,
            address: addr.into(),
            transitions: [].into(),
            matches: [].into(),
        }
    }

    fn root() -> TreeNode {
        Self::with_address(PatternNodeAddress(0, 0))
    }

    fn line_root(line_ind: usize) -> TreeNode {
        Self::with_address(PatternNodeAddress(line_ind, 0))
    }
}

pub struct NaiveManyPatternMatcher {
    tree: PatternTrie,
    patterns: Vec<Pattern>,
}

impl StateAutomatonMatcher for NaiveManyPatternMatcher {
    type StateID = TreeNodeID;
    type Address = PatternNodeAddress;

    fn root(&self) -> (Self::StateID, Self::Address) {
        (TreeNodeID::ROOT, PatternNodeAddress(0, 0))
    }

    fn next_states(
        &self,
        state: &Self::StateID,
        graph: &PortGraph,
        mapped_nodes: &BiBTreeMap<NodeIndex, Self::Address>,
    ) -> Vec<(Self::StateID, Option<NodeIndex>)> {
        let mut next_states = Vec::new();
        let mut state = Some(state.clone());
        while let Some(curr_state) = state {
            // Compute "ideal" transition
            let (mut transition, mut next_node) = self
                .tree
                .compute_transition_for_state(&curr_state, graph, mapped_nodes)
                // Default to Fail transition
                .unwrap_or((NodeTransition::Fail, None));
            // Fall-back to simpler transitions if non-existent
            while self.tree.transition(&curr_state, &transition).is_none()
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
            if let Some(next_state) = self.tree.transition(&curr_state, &transition) {
                next_states.push((next_state.clone(), next_node));
            }
            // Repeat if we are at a root (i.e. non-deterministic)
            if curr_state.ind == 0 {
                state = self
                    .tree
                    .transition(&curr_state, &NodeTransition::Fail)
                    .cloned();
            } else {
                break;
            }
        }
        next_states
    }

    fn visit_node(
        &self,
        mapped_nodes: &mut BiBTreeMap<NodeIndex, Self::Address>,
        state: &Self::StateID,
        node: Option<NodeIndex>,
    ) {
        let Some(addr) = &self.tree.node(state).address else {
            return
        };
        let Some(node) = node else {
            return
        };
        if !mapped_nodes.contains_left(&node) {
            mapped_nodes
                .insert_no_overwrite(node, addr.clone())
                .expect("Could not insert address into map");
        }
        if mapped_nodes.get_by_left(&node) != Some(addr) {
            panic!("Could not insert address into map")
        }
    }

    fn matches(&self, state: &Self::StateID) -> Vec<PatternID> {
        self.tree.node(state).matches.clone()
    }
}

impl NaiveManyPatternMatcher {
    pub fn new() -> Self {
        Self {
            tree: PatternTrie::new(),
            patterns: Vec::new(),
        }
    }

    pub fn add_pattern(&mut self, pattern: Pattern) -> Option<PatternID> {
        let pattern_id = self.tree.add_pattern(&pattern);
        self.patterns.push(pattern);
        pattern_id
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use itertools::Itertools;
    use portgraph::{Direction, NodeIndex, PortGraph};

    use proptest::prelude::*;

    use crate::{
        matcher::{
            many_patterns::{naive::PatternTrie, PatternID, PatternMatch, StateAutomatonMatcher},
            Matcher, SinglePatternMatcher,
        },
        pattern::{Edge, Pattern},
        utils::test_utils::{arb_portgraph_connected, graph, non_empty_portgraph},
    };

    use super::{
        NaiveManyPatternMatcher, NodeTransition, PatternNodeAddress, TreeNode, TreeNodeID,
    };

    #[test]
    fn a_few_patterns() {
        let mut trie = PatternTrie::new();

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(1, 1);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        let pattern = Pattern::from_graph(g.clone()).unwrap();
        trie.add_pattern(&pattern);

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
        trie.add_pattern(&pattern);

        let mut g = PortGraph::new();
        let v0 = g.add_node(0, 1);
        let v1 = g.add_node(2, 0);
        let v0_out0 = g.port(v0, 0, portgraph::Direction::Outgoing).unwrap();
        let v1_in0 = g.port(v1, 0, portgraph::Direction::Incoming).unwrap();
        g.link_ports(v0_out0, v1_in0).unwrap();
        let pattern = Pattern::from_graph(g.clone()).unwrap();
        trie.add_pattern(&pattern);

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
        trie.add_pattern(&pattern);

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
        trie.add_pattern(&pattern);

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
        trie.add_pattern(&pattern);

        let expected_trie = fs::read_to_string("patterntrie_baby.gv").unwrap();
        assert_eq!(expected_trie, trie.dotstring());
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

        fs::write("debug.gv", matcher.tree.dotstring()).unwrap();

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
        fs::write("patterntrie.gv", matcher.tree.dotstring());
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
        #[test]
        fn single_graph_proptest(pattern in arb_portgraph_connected(10, 4, 20), g in non_empty_portgraph(100, 4, 200)) {
            let pattern = Pattern::from_graph(pattern).unwrap();
            let mut matcher = NaiveManyPatternMatcher::new();
            let pattern_id = matcher.add_pattern(pattern.clone()).unwrap();
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
        #[test]
        fn many_graphs_proptest(
            patterns in prop::collection::vec(arb_portgraph_connected(10, 4, 20), 1..10),
            g in non_empty_portgraph(100, 4, 200)
        ) {
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            let mut matcher = NaiveManyPatternMatcher::new();
            for p in patterns.iter() {
                matcher.add_pattern(p.clone()).unwrap();
            }
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
            fs::write("patterntrie.gv", matcher.tree.dotstring());
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
