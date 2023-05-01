#![warn(missing_docs)]

//! Fast pattern matching on port graphs.
//!
//! This crate is designed to find embeddings of port graphs fast, in the case
//! of many patterns. Patterns are preprocessed to make matching faster and
//! compiled into a graph trie data structure similar to the finite automata
//! used for string matching and regular expressions.
//!
//! # Examples
//!
//! ```
//! use portgraph::PortGraph;
//! use portmatching::*;
//! # use portmatching::pattern::InvalidPattern;
//!
//! let (mut g1, mut g2) = (PortGraph::new(), PortGraph::new());
//! g1.add_node(0, 0);
//! g2.add_node(1, 1);
//! let (p1, p2) = (Pattern::from_graph(g1.clone())?, Pattern::from_graph(g2)?);
//! let trie = LineGraphTrie::from_patterns([p1, p2]);
//! trie.find_matches(&g1);
//! # Ok::<(), InvalidPattern>(())
//! ```

pub mod addressing;
pub mod graph_tries;
pub mod matcher;
pub mod pattern;
pub mod utils;

pub use matcher::{LineGraphTrie, Matcher, NaiveManyMatcher, PatternID, SinglePatternMatcher, ManyPatternMatcher};
pub use pattern::Pattern;
