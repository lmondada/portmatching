mod constraint;
pub(crate) mod pattern;
mod portmatcher;
mod weighted_graph;

use constraint::PortgraphConstraint;
use portgraph::{PortGraph, Weights};

pub use self::portmatcher::{PortMatcher, RootedPortMatcher};
pub use pattern::{PortgraphPattern, PortgraphPatternBuilder};
pub use weighted_graph::WeightedPortGraphRef;

use std::hash::Hash;
use std::iter::{self, Map, Repeat, Zip};
use std::ops::RangeFrom;

pub trait EdgeProperty: Clone + Ord + Hash {
    type OffsetID: Eq + Copy;

    fn reverse(&self) -> Option<Self>;

    fn offset_id(&self) -> Self::OffsetID;
}


pub trait NodeProperty: Clone + Hash + Ord {}

impl<U: Clone + Hash + Ord> NodeProperty for U {}

impl<A: Copy + Ord + Hash> EdgeProperty for (A, A) {
    type OffsetID = A;

    fn reverse(&self) -> Option<Self> {
        (self.1, self.0).into()
    }

    fn offset_id(&self) -> Self::OffsetID {
        self.0
    }
}

impl EdgeProperty for () {
    type OffsetID = ();

    fn reverse(&self) -> Option<Self> {
        ().into()
    }

    fn offset_id(&self) -> Self::OffsetID {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Symbol(IterationStatus, usize);
type SymbolsIter =
    Map<Zip<Repeat<IterationStatus>, RangeFrom<usize>>, fn((IterationStatus, usize)) -> Symbol>;

impl Symbol {
    pub(crate) fn new(status: IterationStatus, ind: usize) -> Self {
        Self(status, ind)
    }

    fn from_tuple((status, ind): (IterationStatus, usize)) -> Self {
        Self::new(status, ind)
    }

    pub(crate) fn root() -> Self {
        Self(IterationStatus::Skeleton(0), 0)
    }

    pub(crate) fn symbols_in_status(status: IterationStatus) -> SymbolsIter {
        iter::repeat(status).zip(0..).map(Self::from_tuple)
    }
}

impl crate::Symbol for Symbol {
    fn root() -> Self {
        Symbol::root()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum IterationStatus {
    // We are traversing the i-th line wihtin skeleton
    Skeleton(usize),
    // We are traversing the i-th line outside skeleton
    LeftOver(usize),
    // We are done
    Finished,
}

impl IterationStatus {
    fn increment(&mut self, max_i: usize) {
        *self = match self {
            Self::Skeleton(i) => {
                if *i + 1 < max_i {
                    Self::Skeleton(*i + 1)
                } else {
                    Self::LeftOver(0)
                }
            }
            Self::LeftOver(i) => {
                if *i + 1 < max_i {
                    Self::LeftOver(*i + 1)
                } else {
                    Self::Finished
                }
            }
            Self::Finished => Self::Finished,
        }
    }
}


#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum NodeLocation {
    // The node is in an already-known location
    Exists(Symbol),
    // We need to explore along the i-th line to discover the node
    Discover(usize),
}