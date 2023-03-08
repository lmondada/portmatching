use std::{fmt, collections::BTreeSet};

use bimap::BiBTreeMap;
use portgraph::{NodeIndex, PortGraph, PortIndex};

use crate::pattern::Pattern;

use super::Matcher;

mod naive;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternMatch {
    id: PatternID,
    root: NodeIndex,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PatternID(usize);

impl fmt::Debug for PatternID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

trait StateAutomatonMatcher {
    type StateID;
    type Address: Clone + Ord;

    fn root(&self) -> (Self::StateID, Self::Address);

    fn next_states(
        &self,
        state: &Self::StateID,
        graph: &PortGraph,
        mapped_nodes: &BiBTreeMap<NodeIndex, Self::Address>,
    ) -> Vec<(Self::StateID, Option<NodeIndex>)>;

    fn visit_node(
        &self,
        mapped_nodes: &mut BiBTreeMap<NodeIndex, Self::Address>,
        state: &Self::StateID,
        node: Option<NodeIndex>,
    );

    fn matches(&self, node: &Self::StateID) -> Vec<PatternID>;
}

impl<A: StateAutomatonMatcher> Matcher for A {
    type Match = PatternMatch;

    fn find_anchored_matches(&self, graph: &PortGraph, root: NodeIndex) -> Vec<Self::Match> {
        let (root_state, root_address) = self.root();
        let mut curr_states =
            Vec::from([(root_state, BiBTreeMap::from_iter([(root, root_address)]))]);
        let mut matches = BTreeSet::new();
        while !curr_states.is_empty() {
            let mut new_curr_states = Vec::new();
            for (state, mapped_nodes) in curr_states {
                for pattern_id in self.matches(&state) {
                    matches.insert(PatternMatch {
                        id: pattern_id,
                        root,
                    });
                }
                for (next_state, next_node) in self.next_states(&state, graph, &mapped_nodes) {
                    let mut mapped_nodes = mapped_nodes.clone();
                    self.visit_node(&mut mapped_nodes, &next_state, next_node);
                    new_curr_states.push((next_state, mapped_nodes));
                }
            }
            curr_states = new_curr_states;
        }
        Vec::from_iter(matches)
    }
}
