# portmatching

[![build_status][]](https://github.com/lmondada/portmatching/actions)
[![msrv][]](https://github.com/lmondada/portmatching)

Fast pattern matching on port graphs.

This crate is designed to find embeddings of port graphs fast, in the case
of many patterns. Patterns are preprocessed to make matching faster and
compiled into a graph trie data structure similar to the finite automata
used for string matching and regular expressions.

The crate exports several [`Matcher`](crate::Matcher) objects that can be used for matching. In order of complexity:
-   [`SinglePatternMatcher`](crate::SinglePatternMatcher): Only supports matching a single pattern at the time, corresponds as is typically done
-   [`NaiveManyMatcher`](crate::NaiveManyMatcher): Obtained by combining multiple
[`SinglePatternMatcher`](crate::SinglePatternMatcher) together, matching one pattern at at time. This is the baseline for benchmarking
-   [`NonDetTrieMatcher`](crate::NonDetTrieMatcher): A naive matcher similar to [`NaiveManyMatcher`](crate::NaiveManyMatcher) but stored as a graph trie
-   [`DetTrieMatcher`](crate::DetTrieMatcher): The "optimal" graph trie, in the sense
that a minimum number of states will be traversed in the finite automaton at pattern
matching time. However, the size (and construction cost) of the trie will scale badly.
-   [`BalancedTrieMatcher`](crate::BalancedTrieMatcher): A compromise between
[`NonDetTrieMatcher`](crate::NonDetTrieMatcher) and [`DetTrieMatcher`](crate::DetTrieMatcher) that is expected to combine the benefits of the two strategies, especially in the case of circuit-like graphs.

## Example

```
use portgraph::{PortGraph, PortMut};
use portmatching::*;

let (mut g1, mut g2) = (PortGraph::new(), PortGraph::new());
g1.add_node(0, 0);
g2.add_node(1, 1);
let (p1, p2) = (UnweightedPattern::from_portgraph(&g1), UnweightedPattern::from_portgraph(&g2));
let trie = ManyMatcher::from_patterns(vec![p1, p2]);
trie.find_matches(&g1);
```

## Features

Note: none of these features currently offer useful features to the end user of
the crate. They are useful for testing and benchmarking.

-   `serde`: Enable serialization and deserialization via serde. Currently WIP.
-   `datagen`: Necessary for the [`data_generation`](src/bin/data_generation.rs) binary.

## License

Distributed under the MIT License. See [LICENSE][] for more information.

  [build_status]: https://github.com/lmondada/portmatching/workflows/Continuous%20integration/badge.svg?branch=main
  [LICENSE]: LICENCE
  [msrv]: https://img.shields.io/badge/rust-1.70.0%2B-blue.svg?maxAge=3600
