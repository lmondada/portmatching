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
//! use portmatching::{LineGraphTrie, Pattern};
//!
//! let (mut g1, mut g2) = (PortGraph::new(), PortGraph::new());
//! let (p1, p2) = (Pattern::from_graph(g1), Pattern::from_graph(g2));
//! let trie = LineGraphTrie::from_patterns([p1, p2]);
//! trie.find_matches(&g1);
//! ```

pub mod addressing;
pub mod graph_tries;
pub mod matcher;
pub mod pattern;
mod utils;

pub use matcher::{LineGraphTrie, Matcher, NaiveManyMatcher, PatternID, SinglePatternMatcher};
pub use pattern::Pattern;
