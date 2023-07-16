# portmatching

[![build_status][]](https://github.com/lmondada/portmatching/actions)
[![msrv][]](https://github.com/lmondada/portmatching)

Fast pattern matching on port graphs.

This crate is designed to find embeddings of port graphs fast, in the case
of many patterns. Patterns are preprocessed to make matching faster and
compiled into a graph trie data structure similar to the finite automata
used for string matching and regular expressions.

The crate exports several [`Matcher`](crate::Matcher) instances that can be used for pattern matching.
The principal object of interest in this crate is the [`ManyMatcher`](crate::ManyMatcher) object. 
The other [`Matcher`](crate::Matcher)s serve as baselines for benchmarking purposes.
[`SinglePatternMatcher`](crate::SinglePatternMatcher) matches a single pattern
in time `O(nm)` (size of the input times size of the pattern).
The [`NaiveManyMatcher`](crate::NaiveManyMatcher) uses `k` instances of
the [`SinglePatternMatcher`](crate::SinglePatternMatcher) to find matches
of any of `k` patterns in time `O(kmn)`.

## Benchmarks

#### Comparison to baseline
Pattern matching times for 0 ... 1000 patterns, `NaiveManyMatcher` vs automaton-based `ManyMatcher`.

![comparison with baseline](benches/many_matchers.svg)

#### Pattern matching scaling (on-line)
This plot measures the time (ms) it takes to perform matches for `k` patterns,
as a function of `k`.
Patterns and input are random graphs that are quantum circuit-like.
Weights are chosen at random.
The input graph has 2000 nodes, patterns have between 2 and 5 qubits and up to 30 nodes.

![pattern matching as a fn of patterns](benches/pattern_scaling.svg)

#### Automaton construction time (off-line)
On top of the running time plotted above, there is also a one-time cost to
construct the automaton from the set of patterns.
This is plotted here, again as a function of the number of patterns.

![trie construction times](benches/trie_construction.svg)

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

-   `serde`: Enable serialization and deserialization via serde.
-   `datagen`: Necessary for the [`data_generation`](src/bin/data_generation.rs) binary, for benchmarking. Currently not useful to the end user of
the crate.


## License

Distributed under the MIT License. See [LICENSE][] for more information.

  [build_status]: https://github.com/lmondada/portmatching/workflows/Continuous%20integration/badge.svg?branch=main
  [LICENSE]: LICENCE
  [msrv]: https://img.shields.io/badge/rust-1.70.0%2B-blue.svg?maxAge=3600
