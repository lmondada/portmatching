use std::fmt::{self, Debug, Display};

use portgraph::{dot::dot_string_weighted, Direction, NodeIndex, PortGraph, PortIndex, Weights};

use crate::constraint::Constraint;

use super::{GraphTrie, GraphTrieBuilder, StateID};

/// A node in the GraphTrie.
///
/// The pair `port_offset` and `address` indicate the next edge to follow.
/// The `find_port` serves to store the map NodeTransition => PortIndex.
///
/// `port_offset` and `address` can be unset (ie None), in which case the
/// transition Fail is the only one that should be followed. At write time,
/// an unset field is seen as a license to assign whatever is most convenient.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct NodeWeight<A> {
    pub(crate) out_port: Option<A>,
    pub(crate) non_deterministic: bool,
}

impl<A> Default for NodeWeight<A> {
    fn default() -> Self {
        Self {
            out_port: Default::default(),
            non_deterministic: Default::default(),
        }
    }
}

impl<A: Debug> Display for NodeWeight<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.non_deterministic {
            write!(f, "<font color=\"red\">")?;
        }
        if let Some(port) = &self.out_port {
            write!(f, "[{port:?}]")?;
        }
        if self.non_deterministic {
            write!(f, "</font>")?;
        }
        Ok(())
    }
}

// impl<'n, T: SpineAddress> NodeWeight<T> {
//     fn address(&'n self) -> Option<Address<T::AsRef<'n>>> {
//         let (spine_ind, ind) = self.address?;
//         let spine = self.spine.as_ref()?[spine_ind].as_ref();
//         Some((spine, ind))
//     }
// }

pub(super) type EdgeWeight<C> = Option<C>;

/// A graph trie implementation using portgraph.
///
/// The parameter T is the type of the spine. Any type that implements
/// [`SpineAddress`] can be used as spine in theory, however constructing such
/// a trie is only supported for very specific types.
///
/// The construction of such tries is the most complex logic in this
/// crate.
///
/// There are deterministic and non-deterministic states, describing how
/// the state automaton in that state should behave.
/// Roughly speaking, the role of deterministic and non-deterministic states is
/// inverted when writing to the trie (as opposed to reading):
///  * a deterministic state at read time means that every possible transition must
///    be considered at write time (as we do not know which one will be taken)
///  * a non-deterministic state at read time on the other hand is guaranteed to
///    follow any allowed transition, and thus at write time we can just follow
///    one of the transitions.
///
/// The deterministic states are thus the harder states to handle at write time.
/// The main idea is that transitions are considered from left to right, so when
/// adding a new transition, all the transitions to its right might get
/// "shadowed" by it. That means that we must make sure that if an input graph
/// chooses the new transition over the old one, then all the patterns that would
/// have been discovered along the old path will still be discovered.
///
/// Similarly, transitions to the left of a new transition can shadow the new
/// transition, so we must make sure that if an input graph chooses one of these
/// higher priority transition, it still discovers the pattern that is being added.
///
/// As transitions are added we also keep track of how the trie is traversed,
/// forming the "trace" of the trie. This is used in [`Self::finalize`] to
/// split states where necessary to keep avoid cross-talk, i.e. creating new
/// disallowed state transitions as a by-product of the new transitions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BaseGraphTrie<C, A> {
    pub(crate) graph: PortGraph,
    pub(crate) weights: Weights<NodeWeight<A>, EdgeWeight<C>>,
}

impl<C, A> Debug for BaseGraphTrie<C, A>
where
    A: Debug,
    C: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaseGraphTrie")
            .field("graph", &self.graph)
            .field("weights", &self.weights)
            .finish()
    }
}

impl<C: Clone, A: Clone> Clone for BaseGraphTrie<C, A> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            weights: self.weights.clone(),
        }
    }
}

impl<C: Clone + Ord + Constraint, A: Clone + Ord> Default for BaseGraphTrie<C, A> {
    fn default() -> Self {
        let graph = Default::default();
        let weights = Default::default();
        let mut ret = Self { graph, weights };
        ret.add_state(true);
        ret
    }
}

impl<C: Clone, A: Clone> GraphTrie for BaseGraphTrie<C, A> {
    type Constraint = C;
    type Address = A;

    fn trie(&self) -> &PortGraph {
        &self.graph
    }

    fn port_address(&self, state: StateID) -> Option<&A> {
        self.weights[state].out_port.as_ref()
    }

    fn transition(&self, port: PortIndex) -> Option<&Self::Constraint> {
        self.weights[port].as_ref()
    }

    fn is_non_deterministic(&self, state: StateID) -> bool {
        self.weights[state].non_deterministic
    }
}

impl<C: Display + Clone, A: Debug + Clone> BaseGraphTrie<C, A> {
    pub(crate) fn str_weights(&self) -> Weights<String, String> {
        let mut str_weights = Weights::new();
        for p in self.graph.ports_iter() {
            str_weights[p] = match self.graph.port_direction(p).unwrap() {
                Direction::Incoming => "".to_string(),
                Direction::Outgoing => self.weights[p]
                    .as_ref()
                    .map(|c| c.to_string())
                    .unwrap_or("FAIL".to_string()),
            };
        }
        for n in self.graph.nodes_iter() {
            str_weights[n] = self.weights[n].to_string();
        }
        str_weights
    }

    pub(crate) fn _dotstring(&self) -> String {
        dot_string_weighted(&self.graph, &self.str_weights())
    }
}

impl<C: Clone + Ord + Constraint, A: Clone + Ord> BaseGraphTrie<C, A> {
    /// An empty graph trie
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to convert into a non-deterministic state.
    ///
    /// If the state is already non-deterministic, this is a no-op. If it is
    /// deterministic and has more than one transition, then the conversion
    /// fails.
    ///
    /// Returns whether the state is non-deterministic after the conversion.
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn into_non_deterministic(&mut self, state: NodeIndex) -> bool {
        let det_flag = &mut self.weights[state].non_deterministic;
        if self.graph.num_outputs(state) <= 1 {
            *det_flag = true;
        }
        *det_flag
    }

    pub(crate) fn weight(&self, state: StateID) -> &NodeWeight<A> {
        &self.weights[state]
    }

    /// View the trie as a builder, for constructions
    pub fn as_builder<Age: Default + Clone>(self) -> GraphTrieBuilder<C, A, Age> {
        GraphTrieBuilder::new(self)
    }

    pub(super) fn add_state(&mut self, non_deterministic: bool) -> StateID {
        let node = self.graph.add_node(0, 0);
        self.weights[node].non_deterministic = non_deterministic;
        node
    }

    /// Try to convert into a start state for `graph_edge`
    #[allow(clippy::wrong_self_convention)]
    pub(super) fn into_start_state(
        &mut self,
        trie_state: StateID,
        out_port: &A,
        deterministic: bool,
    ) -> bool {
        // let start_node = graph.port_node(graph_edge).expect("invalid port");
        // let start_offset = graph.port_offset(graph_edge).expect("invalid port");

        let trie_out_port = self
            .weight(trie_state)
            .out_port
            .as_ref()
            .unwrap_or(out_port);
        if trie_out_port == out_port {
            self.weights[trie_state].out_port = Some(trie_out_port.clone());
            if !deterministic {
                // Try to convert state into a non-deterministic one
                self.into_non_deterministic(trie_state);
            }
            true
        } else {
            false
        }
    }

    pub(super) fn set_num_ports(&mut self, state: StateID, incoming: usize, outgoing: usize) {
        // if state == NodeIndex::new(2) {
        // }
        self.graph
            .set_num_ports(state, incoming, outgoing, |old, new| {
                let new = new.new_index();
                self.weights.ports.rekey(old, new);
            });
    }
}
