#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

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
pub mod matcher;
pub mod pattern;
pub mod graph_tries;
mod utils;

pub use matcher::{LineGraphTrie, NaiveManyMatcher, SinglePatternMatcher, Matcher, PatternID};
pub use pattern::Pattern;