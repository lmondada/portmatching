use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{self, Debug},
};

use bimap::BiBTreeMap;
use portgraph::{NodeIndex, PortGraph};

use crate::{pattern::Edge, PortOffset};

use super::{ManyPatternMatcher, ReadGraphTrie, WriteGraphTrie};

/// Every node in a pattern has an address, given by the line index and
/// the index within the line
///
/// An address is unique within a pattern, and will always be unique within
/// a maximum match as well
///
/// Note that a same node in a graph may have more than one address. In this
/// case, we always refer to the node by its smallest address, so that the
/// map address <-> node ID is bijective
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternNodeAddress(usize, usize);

impl PatternNodeAddress {
    /// The root of the pattern is always assigned the address (0, 0)
    const ROOT: PatternNodeAddress = Self(0, 0);

    /// The next address on the same line
    fn next(&self) -> Self {
        Self(self.0, self.1 + 1)
    }

    /// The first (root) address on the next line
    fn next_root(&self) -> Self {
        Self(self.0 + 1, 0)
    }
}

impl Debug for PatternNodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", (self.0, self.1))
    }
}

/// A unique ID for every node in the NaiveGraphTrie
///
/// The 2D indices into the nodes vector of the NaiveGraphTrie
///
/// Similar in flavour, but not be confused with the address! An address may
/// appear in multiple nodes in the graph tree, hence with different unique ID.
///
/// The point of addresses, however, is that two nodes with the same address
/// will never be match at the same time.
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

    /// The root of the trie, at [0, 0]
    const ROOT: TreeNodeID = TreeNodeID {
        line_tree: 0,
        ind: 0,
    };
}

/// All possible transitions in the Graph trie
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum NodeTransition {
    /// Following the current edge leads to a known node, with the given addr
    /// and coming in through `port`
    KnownNode(PatternNodeAddress, PortOffset),
    /// Following the current edge leads to an unknown node, and coming in
    /// through `port`
    NewNode(PortOffset),
    /// The current edge is not linked to any other vertex
    NoLinkedNode,
    /// The current edge does not exist. For root vertices in the trie,
    /// fail transitions also indicate an epsilon transition
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

/// A node in the NaiveGraphTrie
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `transition` map indicates the states that can be transitioned to,
/// according to what the next edge leads to.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, Default)]
struct TreeNode {
    port_offset: Option<PortOffset>,
    address: Option<PatternNodeAddress>,
    transitions: BTreeMap<NodeTransition, TreeNodeID>,
}

/// A simple implementation of a graph trie
///
/// The graph trie is organised in "lines". Within each line (corresponding
/// to a maximal line of the patterns), the node transitions are deterministic.
/// Between lines, transitions are non-deterministic, so that more than one
/// line may have to be traversed concurrently at any one time.
pub struct NaiveGraphTrie(Vec<Vec<TreeNode>>);

impl Default for NaiveGraphTrie {
    fn default() -> Self {
        Self(vec![vec![TreeNode::root()]])
    }
}

#[derive(PartialEq, Eq)]
enum TrieTraversal {
    ReadOnly,
    Write,
}

impl NaiveGraphTrie {
    /// An empty graph trie
    pub fn new() -> Self {
        Self::default()
    }

    /// Follow a transition from `node` along `transition`
    ///
    /// Returns None if the transition does not exist
    fn transition(&self, node: TreeNodeID, transition: &NodeTransition) -> Option<TreeNodeID> {
        self.state(node).transitions.get(transition).copied()
    }

    /// Set a new transition between `node` and `next_node`
    ///
    /// If the transition already exists, then trying to update it to a
    /// different value will return None. In most cases this should be
    /// considered a logic mistake.
    #[must_use]
    fn set_transition(
        &mut self,
        node: TreeNodeID,
        next_node: TreeNodeID,
        transition: &NodeTransition,
    ) -> Option<TreeNodeID> {
        match transition {
            NodeTransition::NewNode(_) | NodeTransition::KnownNode(_, _) => {
                assert!(node.ind < next_node.ind);
            }
            NodeTransition::Fail | NodeTransition::NoLinkedNode => {
                assert!(node.line_tree < next_node.line_tree);
                assert_eq!(next_node.ind, 0);
            }
        }
        let old = self
            .state_mut(node)
            .transitions
            .insert(transition.clone(), next_node);
        if old.is_some() && old != Some(next_node) {
            None
        } else {
            self.transition(node, transition)
        }
    }

    /// The node at address `node`
    fn state(&self, node: TreeNodeID) -> &TreeNode {
        &self.0[node.line_tree][node.ind]
    }

    /// A mutable reference to the `node` state
    fn state_mut(&mut self, node: TreeNodeID) -> &mut TreeNode {
        &mut self.0[node.line_tree][node.ind]
    }

    /// Copy all data from `node` into `node_to`
    pub fn clone_into(&mut self, node: TreeNodeID, node_to: TreeNodeID) {
        self.state_mut(node_to).port_offset = self.state(node).port_offset.clone();
        self.state_mut(node_to).address = self.state(node).address.clone();
    }

    /// Add a new state to an exisiting tree
    fn append_tree(&mut self, tree_ind: usize) -> TreeNodeID {
        let ind = self.0[tree_ind].len();
        self.0[tree_ind].push(TreeNode::new());
        TreeNodeID::new(tree_ind, ind)
    }

    /// Add a new tree to the graph trie
    fn add_new_tree(&mut self) -> TreeNodeID {
        let node_id = TreeNodeID::line_root(self.0.len());
        self.0.push(vec![TreeNode::new()]);
        node_id
    }

    /// Extend an existing tree to handle a longer maximal line
    ///
    /// This is in principle similar to `append_tree`, except that it preserves
    /// all the "business logic" of the graph trie.
    /// That is to say, it will do a lot of nodes copying so that any patterns
    /// that were previously added to the trie will still match along the newly
    /// added extension
    ///
    /// This is to be refactored and cleaned up -- there is a lot going on here.
    fn extend_tree<F: FnMut(&TreeNodeID, &TreeNodeID)>(
        &mut self,
        node: TreeNodeID,
        addr: PatternNodeAddress,
        clone_state: &mut F,
    ) -> TreeNodeID {
        let next_node = self.append_tree(node.line_tree);
        self.state_mut(next_node).address = addr.clone().into();
        let fallback = if Self::is_root(node) {
            self.transition(node, &NodeTransition::NoLinkedNode)
        } else {
            self.transition(node, &NodeTransition::NoLinkedNode)
                .or_else(|| self.transition(node, &NodeTransition::Fail))
        };
        if let Some(fallback) = fallback {
            // For each NewNode(port) transition in fallback, we need to add the
            // KnownNode(addr, port) transition for the new addr
            let subtree = self.clone_subtree(fallback, clone_state);
            for node in self.all_nodes_in(subtree) {
                let new_node_transitions: Vec<_> = self
                    .state(node)
                    .transitions
                    .iter()
                    .filter_map(|(label, subtree)| {
                        if let NodeTransition::NewNode(port) = label {
                            Some((port.clone(), *subtree))
                        } else {
                            None
                        }
                    })
                    .collect();
                for (port, next_node) in new_node_transitions {
                    let new_subtree = self.clone_subtree(next_node, clone_state);
                    if self
                        .set_transition(
                            node,
                            new_subtree,
                            &NodeTransition::KnownNode(addr.clone(), port.clone()),
                        )
                        .is_some()
                    {
                        if let Some(old_addr) = self.state(next_node).address.clone() {
                            for node in self.all_nodes_in(new_subtree) {
                                let node = self.state_mut(node);
                                // change the address of node
                                if node.address.is_some()
                                    && node.address.as_ref().unwrap() == &old_addr
                                {
                                    node.address = Some(addr.clone());
                                }
                                // change the transitions
                                if let Some(transition) = node.transitions.remove(
                                    &NodeTransition::KnownNode(old_addr.clone(), port.clone()),
                                ) {
                                    node.transitions.insert(
                                        NodeTransition::KnownNode(addr.clone(), port.clone()),
                                        transition,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            self.set_transition(next_node, subtree, &NodeTransition::NoLinkedNode)
                .expect("Changing existing transition");
            let subtree = self.clone_subtree(fallback, clone_state);
            self.set_transition(next_node, subtree, &NodeTransition::Fail)
                .expect("Changing existing transition");
        }
        next_node
    }

    /// Whether coordinates are at the beginning of a tree
    fn is_root(root: TreeNodeID) -> bool {
        root.ind == 0
    }

    /// A vector of all nodes that are descendants of `root`
    fn all_nodes_in(&self, root: TreeNodeID) -> Vec<TreeNodeID> {
        // Add new trees
        let mut line_trees = VecDeque::new();
        let mut nodes = Vec::new();
        if Self::is_root(root) {
            line_trees.push_back(root.line_tree);
        } else {
            let mut curr_nodes = VecDeque::from([root]);
            while let Some(node) = curr_nodes.pop_front() {
                for (label, next_node) in self.state(node).transitions.clone() {
                    match label {
                        NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                            curr_nodes.push_back(next_node);
                        }
                        NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                            line_trees.push_back(next_node.line_tree);
                        }
                    }
                }
                nodes.push(node);
            }
        }
        while let Some(line_tree) = line_trees.pop_front() {
            // Iterate through line_tree
            for (ind, node) in self.0[line_tree].iter().enumerate() {
                for (label, next_node) in node.transitions.iter() {
                    match label {
                        NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                            line_trees.push_back(next_node.line_tree);
                        }
                        _ => {}
                    }
                }
                nodes.push(TreeNodeID { line_tree, ind });
            }
        }
        nodes
    }

    /// Clone everything from `old_root` and descendents
    ///
    /// If `old_root` is the root of a subtree, then the new
    /// root will be in a new subtree.
    /// Otherwise, the new root will be in the same subree as `old_root`.
    fn clone_subtree<F: FnMut(&TreeNodeID, &TreeNodeID)>(
        &mut self,
        old_root: TreeNodeID,
        clone_state: &mut F,
    ) -> TreeNodeID {
        let mut line_trees = VecDeque::new();
        let new_root = if Self::is_root(old_root) {
            line_trees.push_back(old_root.line_tree);
            TreeNodeID::new(self.0.len(), 0)
        } else {
            // Clone nodes in current tree
            let mut curr_tree_nodes = VecDeque::from([old_root]);
            let mut first_new_node = None;
            while let Some(old_node) = curr_tree_nodes.pop_front() {
                let new_node = self.append_tree(old_node.line_tree);
                if first_new_node.is_none() {
                    first_new_node = new_node.into();
                }
                // Clone over node
                self.clone_into(old_node, new_node);
                clone_state(&old_node, &new_node);
                // Update transitions
                let n_line_trees = self.0[old_node.line_tree].len();
                let n_trees = self.0.len();
                for (label, next_node) in self.state(old_node).transitions.clone() {
                    match label {
                        transition @ NodeTransition::KnownNode(_, _)
                        | transition @ NodeTransition::NewNode(_) => {
                            self.set_transition(
                                new_node,
                                TreeNodeID {
                                    line_tree: next_node.line_tree,
                                    ind: n_line_trees + curr_tree_nodes.len(),
                                },
                                &transition,
                            )
                            .expect("Changing existing value");
                            curr_tree_nodes.push_back(next_node);
                        }
                        transition @ NodeTransition::NoLinkedNode
                        | transition @ NodeTransition::Fail => {
                            self.set_transition(
                                new_node,
                                TreeNodeID {
                                    line_tree: n_trees + line_trees.len(),
                                    ind: 0,
                                },
                                &transition,
                            )
                            .expect("Changing existing value");
                            line_trees.push_back(next_node.line_tree);
                        }
                    }
                }
            }
            first_new_node.expect("The loop above must execute once")
        };

        // Add new trees
        while let Some(old_line_tree) = line_trees.pop_front() {
            let curr_line_tree = self.0.len();
            // Clone tree
            self.0.push(self.0[old_line_tree].clone());
            // Notify Matcher of clone
            for ind in 0..self.0.len() {
                clone_state(
                    &TreeNodeID {
                        line_tree: old_line_tree,
                        ind,
                    },
                    &TreeNodeID {
                        line_tree: curr_line_tree,
                        ind,
                    },
                );
            }
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

    fn get_transition(
        &self,
        state: TreeNodeID,
        graph: &PortGraph,
        current_match: &MatchObject,
        traversal: TrieTraversal,
    ) -> Vec<(NodeTransition, MatchObject)> {
        // The transition we would want to perform
        let ideal_transition = || {
            let addr = self.state(state).address.as_ref()?;
            // fs::write("patterntrie.gv", self.dotstring()).unwrap();
            // dbg!(&addr);
            // dbg!(&current_match.map);
            let graph_node = *current_match
                .map
                .get_by_right(addr)
                .expect("Malformed pattern trie");
            let out_port = self.state(state).port_offset.clone()?;
            let out_port = out_port.get_index(graph_node, graph)?;
            let in_port = graph.port_link(out_port);

            Some(match in_port {
                Some(in_port) => {
                    let port_offset = PortOffset::try_from_index(in_port, graph).unwrap();
                    let in_node = graph.port_node(in_port).unwrap();
                    match current_match.map.get_by_left(&in_node).cloned() {
                        Some(new_addr) => NodeTransition::KnownNode(new_addr, port_offset),
                        None => NodeTransition::NewNode(port_offset),
                    }
                }
                None => NodeTransition::NoLinkedNode,
            })
        };
        let ideal_transition = ideal_transition().unwrap_or(NodeTransition::Fail);

        // From the ideal transition, compute all valid transitions
        // (depending on whether the trie traversal is read only or write)
        let all_transitions = if traversal == TrieTraversal::ReadOnly {
            // Fall-back to simpler transitions if non-existent
            let mut transition = ideal_transition;
            while self.transition(state, &transition).is_none()
                && transition != NodeTransition::Fail
            {
                transition = match transition {
                    NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                        NodeTransition::NoLinkedNode
                    }
                    NodeTransition::NoLinkedNode => NodeTransition::Fail,
                    NodeTransition::Fail => unreachable!(),
                };
            }
            vec![transition]
        } else {
            let mut all_transitions = Vec::new();
            if let NodeTransition::NewNode(port) = &ideal_transition {
                all_transitions.extend(
                    current_match
                        .no_map
                        .iter()
                        .map(|addr| NodeTransition::KnownNode(addr.clone(), port.clone())),
                );
            }
            all_transitions.push(ideal_transition);
            all_transitions
        };

        // Return transitions, in tuple with update match objects
        all_transitions
            .into_iter()
            .map(|transition| {
                let mut next_match = current_match.clone();
                let in_port = match &transition {
                    NodeTransition::Fail => {
                        if !Self::is_root(state) {
                            let next_addr = current_match.current_addr.next_root();
                            next_match.current_addr = next_addr;
                        }
                        return (transition, next_match);
                    }
                    NodeTransition::NoLinkedNode => {
                        let next_addr = current_match.current_addr.next_root();
                        next_match.current_addr = next_addr;
                        return (transition, next_match);
                    }
                    NodeTransition::KnownNode(_, _) | NodeTransition::NewNode(_) => {
                        let addr = &self.state(state).address.as_ref().unwrap();
                        let graph_node = *current_match
                            .map
                            .get_by_right(addr)
                            .expect("Malformed pattern trie");
                        let out_port = self.state(state).port_offset.clone().unwrap();
                        let out_port = out_port.get_index(graph_node, graph).unwrap();
                        next_match.current_addr = current_match.current_addr.next();
                        graph.port_link(out_port).unwrap()
                    }
                };

                let next_graph_node = graph.port_node(in_port).expect("Invalid port");
                let next_addr = match &transition {
                    NodeTransition::KnownNode(addr, _) => addr.clone(),
                    NodeTransition::NewNode(_) => next_match.current_addr.clone(),
                    NodeTransition::NoLinkedNode | NodeTransition::Fail => {
                        unreachable!()
                    }
                };

                // If we are encountering a known node, but that node was only known
                // in no map, we need to remove it from that map before inserting it
                // into the match map
                next_match.no_map.remove(&next_addr);
                next_match
                    .map
                    .insert_no_overwrite(next_graph_node, next_addr.clone())
                    .ok()
                    .or_else(|| {
                        (next_match.map.get_by_left(&next_graph_node) == Some(&next_addr))
                            .then_some(())
                    })
                    .expect("Map is not injective");
                (transition, next_match)
            })
            .collect()
    }

    /// Assign `node` to given address and port, if possible
    ///
    /// Will update the node fields to the given values if those are None.
    /// Returns whether it was successful
    fn try_update_node(
        &mut self,
        node: TreeNodeID,
        addr: &PatternNodeAddress,
        port_index: &PortOffset,
    ) -> bool {
        let node = self.state_mut(node);
        let res_port_index = node.port_offset.get_or_insert(port_index.clone());
        let res_addr = node.address.get_or_insert(addr.clone());
        res_port_index == port_index && res_addr == addr
    }

    /// Adds NoLinkedNode and Fail transitions
    ///
    /// These link to the transitions of the previous state
    fn add_default_transitions(&mut self, current_state: TreeNodeID, add_fail_link: bool) {
        if !self
            .state(current_state)
            .transitions
            .contains_key(&NodeTransition::NoLinkedNode)
        {
            let new_root = self.add_new_tree();
            self.state_mut(current_state)
                .transitions
                .insert(NodeTransition::NoLinkedNode, new_root);
        }
        if !self
            .state(current_state)
            .transitions
            .contains_key(&NodeTransition::Fail)
            && add_fail_link
        {
            let new_root = self.add_new_tree();
            self.state_mut(current_state)
                .transitions
                .insert(NodeTransition::Fail, new_root);
        }
    }

    /// Finds all possible new lines to transition to
    ///
    /// When inserting "lines" from a pattern into the graph trie, at every
    /// new line one must find all possible transitions that lead to new line;
    /// the "carriage return" operation.
    fn carriage_return(
        &mut self,
        state: TreeNodeID,
        current_match: MatchObject,
        is_dangling: bool,
    ) -> Vec<(TreeNodeID, MatchObject)> {
        let mut current_states = vec![(state, current_match)];
        let mut next_states = Vec::new();
        assert!(!Self::is_root(state) || is_dangling);

        while let Some((current_state, current_match)) = current_states.pop() {
            // Insert empty NoSuchLink and Fail transitions if they do not exist
            self.add_default_transitions(current_state, !is_dangling || current_state != state);
            for (transition, next_state) in &self.state(current_state).transitions {
                match transition {
                    NodeTransition::KnownNode(_, _) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        current_states.push((*next_state, next_match));
                    }
                    NodeTransition::NewNode(_) => {
                        let next_addr = current_match.current_addr.next();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr.clone();
                        next_match.no_map.insert(next_addr);
                        current_states.push((*next_state, next_match));
                    }
                    NodeTransition::NoLinkedNode => {
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((*next_state, next_match));
                    }
                    NodeTransition::Fail => {
                        if is_dangling && current_state == state {
                            continue;
                        }
                        let next_addr = current_match.current_addr.next_root();
                        let mut next_match = current_match.clone();
                        next_match.current_addr = next_addr;
                        next_states.push((*next_state, next_match));
                    }
                }
            }
        }
        next_states
    }

    /// Follow Fail transitions to find state with `out_port`
    fn valid_start_states(
        &mut self,
        (state, current_match): (TreeNodeID, MatchObject),
        out_port: portgraph::PortIndex,
        graph: &PortGraph,
    ) -> Vec<(TreeNodeID, MatchObject)> {
        // We start by finding where we are in the graph, using the outgoing
        // port
        let graph_node = graph.port_node(out_port).expect("Invalid port");
        let graph_port = PortOffset::try_from_index(out_port, graph).expect("Invalid port");

        // Process carriage returns where necessary
        let mut states_post_carriage_return = Vec::new();
        let graph_addr = current_match
            .map
            .get_by_left(&graph_node)
            .expect("Incomplete match map");
        // Follow fail edges until we find the state that
        // corresponds to our position in the graph
        if !self.try_update_node(state, graph_addr, &graph_port) && !Self::is_root(state) {
            states_post_carriage_return.append(&mut self.carriage_return(
                state,
                current_match,
                false,
            ));
        } else {
            states_post_carriage_return.push((state, current_match));
        }

        // Finally we obtain the states where we are meant to be,
        // such that state_addr == edge_addr and state_port == edge_port
        states_post_carriage_return
            .into_iter()
            .map(|(mut state, current_match)| {
                let graph_addr = current_match
                    .map
                    .get_by_left(&graph_node)
                    .expect("Incomplete match map");
                while !self.try_update_node(state, graph_addr, &graph_port) {
                    state = match self.transition(state, &NodeTransition::Fail) {
                        Some(next_node) => next_node,
                        None => {
                            let next_node = self.add_new_tree();
                            self.set_transition(state, next_node, &NodeTransition::Fail)
                                .expect("Changing existing value");
                            next_node
                        }
                    };
                    assert!(Self::is_root(state));
                }
                (state, current_match)
            })
            .collect()
    }

    /// Output trie graph in dotstring format
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

/// An object to store a pattern match during matching
#[derive(Clone, Debug)]
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

impl WriteGraphTrie for NaiveGraphTrie {
    fn create_next_states<F: FnMut(&Self::StateID, &Self::StateID)>(
        &mut self,
        states: Vec<(Self::StateID, Self::MatchObject)>,
        &Edge(out_port, _): &Edge,
        graph: &PortGraph,
        mut clone_state: F,
    ) -> Vec<(Self::StateID, Self::MatchObject)> {
        // Find the states that correspond to out_port
        let states: Vec<_> = states
            .into_iter()
            .flat_map(|state| self.valid_start_states(state, out_port, graph))
            .collect();

        let mut next_states = Vec::new();
        for (state, current_match) in states {
            // For every allowable transition, insert it if it does not exist
            // and return it
            next_states.append(
                &mut self
                    .get_transition(state, graph, &current_match, TrieTraversal::Write)
                    .into_iter()
                    .flat_map(|(transition, next_match)| {
                        if let NodeTransition::NoLinkedNode = &transition {
                            self.carriage_return(state, current_match.clone(), true)
                        } else {
                            [(
                                self.transition(state, &transition).unwrap_or_else(|| {
                                    let next_addr = match &transition {
                                        NodeTransition::KnownNode(addr, _) => addr.clone(),
                                        NodeTransition::NewNode(_) => {
                                            next_match.current_addr.clone()
                                        }
                                        NodeTransition::NoLinkedNode => unreachable!(),
                                        NodeTransition::Fail => {
                                            panic!("transition is not valid")
                                        }
                                    };
                                    let next_state =
                                        self.extend_tree(state, next_addr, &mut clone_state);
                                    self.set_transition(state, next_state, &transition)
                                        .expect("Changing existing value");
                                    next_state
                                }),
                                next_match,
                            )]
                            .into()
                        }
                    })
                    .collect(),
            );
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
        // Compute "ideal" transition
        let mut next_states: Vec<_> = self
            .get_transition(*state, graph, current_match, TrieTraversal::ReadOnly)
            .into_iter()
            .filter_map(|(transition, current_match)| {
                Some((self.transition(*state, &transition)?, current_match))
            })
            .collect();

        // Add epsilon transition if we are at a root (i.e. non-deterministic)
        if state.ind == 0 {
            if let Some(next_state) = self.transition(*state, &NodeTransition::Fail) {
                next_states.push((next_state, current_match.clone()));
            }
        }
        next_states
    }
}

pub type NaiveManyPatternMatcher = ManyPatternMatcher<NaiveGraphTrie>;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use portgraph::{proptest::gen_portgraph, Direction, NodeIndex, PortGraph};

    use proptest::prelude::*;

    use crate::{
        matcher::{
            many_patterns::{naive::NaiveManyPatternMatcher, PatternID, PatternMatch},
            Matcher, SinglePatternMatcher,
        },
        pattern::Pattern,
        utils::test_utils::gen_portgraph_connected,
    };

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
            g.port_index(n, 0, Direction::Outgoing).unwrap(),
            g.port_index(n, 0, Direction::Incoming).unwrap(),
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
            g.port_index(n, 0, Direction::Outgoing).unwrap(),
            g.port_index(n, 0, Direction::Incoming).unwrap(),
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
            g.port_index(n0, 0, Direction::Outgoing).unwrap(),
            g.port_index(n1, 0, Direction::Incoming).unwrap(),
        )
        .unwrap();

        assert_eq!(matcher.find_matches(&g), vec![]);
    }

    #[test]
    fn single_pattern_construction() {
        let mut g = PortGraph::new();
        let n0 = g.add_node(3, 2);
        let n1 = g.add_node(2, 0);
        let n2 = g.add_node(2, 1);
        link(&mut g, (n0, 0), (n0, 2));
        link(&mut g, (n0, 1), (n1, 1));
        link(&mut g, (n2, 0), (n0, 1));
        let p = Pattern::from_graph(g).unwrap();
        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(p);
    }

    fn link(p: &mut PortGraph, (n1, p1): (NodeIndex, usize), (n2, p2): (NodeIndex, usize)) {
        p.link_ports(
            p.port_index(n1, p1, Direction::Outgoing).unwrap(),
            p.port_index(n2, p2, Direction::Incoming).unwrap(),
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
        assert_eq!(matcher.find_matches(&g).len(), 3);
    }

    #[test]
    fn trie_construction_fail() {
        let mut p1 = PortGraph::new();
        let n0 = p1.add_node(2, 1);
        let n1 = p1.add_node(1, 0);
        link(&mut p1, (n0, 0), (n1, 0));
        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 0);
        let n1 = p2.add_node(0, 2);
        link(&mut p2, (n1, 0), (n0, 1));
        link(&mut p2, (n1, 1), (n0, 0));
        NaiveManyPatternMatcher::from_patterns(
            [p1, p2].map(|p| Pattern::from_graph(p).unwrap()).into(),
        );
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
        // fs::write("p1.gv", p1.dotstring()).unwrap();

        let mut p2 = PortGraph::new();
        let n0 = p2.add_node(2, 1);
        let n1 = p2.add_node(3, 0);
        link(&mut p2, (n0, 0), (n1, 1));
        // fs::write("p2.gv", p2.dotstring()).unwrap();

        let mut g = PortGraph::new();
        let n2 = g.add_node(3, 2);
        let n3 = g.add_node(3, 1);
        link(&mut g, (n2, 0), (n3, 1));
        link(&mut g, (n3, 0), (n2, 0));

        let mut matcher = NaiveManyPatternMatcher::new();
        matcher.add_pattern(Pattern::from_graph(p1).unwrap());
        matcher.add_pattern(Pattern::from_graph(p2).unwrap());
        // fs::write("graph.gv", g.dotstring()).unwrap();
        // fs::write("patterntrie.gv", matcher.trie.dotstring()).unwrap();
        assert_eq!(matcher.find_matches(&g).len(), 1);
    }

    proptest! {
        #[test]
        fn single_graph_proptest(pattern in gen_portgraph_connected(10, 4, 20), g in gen_portgraph(100, 4, 200)) {
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
        #[ignore = "very slow"]
        #[test]
        fn many_graphs_proptest(
            patterns in prop::collection::vec(gen_portgraph_connected(10, 4, 20), 1..4),
            g in gen_portgraph(30, 4, 60)
        ) {
            let patterns = patterns
                .into_iter()
                .map(|p| Pattern::from_graph(p).unwrap())
                .collect_vec();
            // fs::write("graph.gv", g.dotstring()).unwrap();
            // for (i, p) in patterns.iter().enumerate() {
            //     fs::write(format!("p{}.gv", i), p.graph.dotstring()).unwrap();
            // }
            let matcher = NaiveManyPatternMatcher::from_patterns(patterns.clone());
            let single_matchers = patterns
                .clone()
                .into_iter()
                .map(SinglePatternMatcher::from_pattern)
                .collect_vec();
            // println!("saved");
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
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v1_0 = g.port_index(v1, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v0_0, v1_0).unwrap();
    //     let v1_1 = g.port_index(v1, 0, Direction::Outgoing).unwrap();
    //     let v3_0 = g.port_index(v3, 0, Direction::Incoming).unwrap();
    //     g.link_ports(v1_1, v3_0).unwrap();
    //     let v3_1 = g.port_index(v3, 0, Direction::Outgoing).unwrap();
    //     let v4_0 = g.port_index(v4, 0, Direction::Incoming).unwrap();
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
    //     let v0_0 = g.port_index(v0, 0, Direction::Outgoing).unwrap();
    //     let v2_1 = g.port_index(v2, 1, Direction::Incoming).unwrap();
    //     let v2_2 = g.port_index(v2, 0, Direction::Outgoing).unwrap();
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
