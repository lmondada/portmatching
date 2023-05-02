//! Caching for addressing schemes.
//!
//! Vertebrae are typically refered to by their canonical path
//! from the root node. This is impractical for caching, as the hashing performance
//! would be poor. Instead, we define a spine ID, a unique vertebra identifier.
//!
//! SpineIDs are chosen to be as small as possible by reusing the same ID when
//! to vertebrae are mutually exclusive (i.e. if they are used in different
//! parts of the trie).
use portgraph::PortOffset;

use super::{Address, NodeOffset};

/// A unique identifier for a vertebra.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default)]
pub struct SpineID(pub(crate) usize);

/// Obtain spine IDs from addresses, for caching.
pub trait AsSpineID {
    /// Get the spine ID of the address, if it exists.
    ///
    /// Returning None will disable caching
    fn as_spine_id(&self) -> Option<SpineID>;
}

impl<'a> AsSpineID for (SpineID, &'a [PortOffset], usize) {
    fn as_spine_id(&self) -> Option<SpineID> {
        Some(self.0)
    }
}

impl<'a> AsSpineID for (&'a [PortOffset], usize) {
    fn as_spine_id(&self) -> Option<SpineID> {
        None
    }
}

/// Cache address computations
///
/// Provide an implementation of [`AddressCache`] to [`super::SkeletonAddressing`] to
/// cache expensive computations.
pub trait AddressCache {
    /// Get the cached node index if it exists.
    fn get_node(&self, _addr: &Address<SpineID>) -> CachedOption<NodeOffset> {
        CachedOption::NoCache
    }

    /// Insert an address to node index map in the cache.
    fn set_node(&mut self, _addr: &Address<SpineID>, _node: NodeOffset) {}
}

impl AddressCache for () {}

/// A simple cache for addresses on the spine.
///
/// Note that this does not cache non-spine addresses.
#[derive(Default)]
pub(crate) struct Cache(Vec<CachedOption<NodeOffset>>);

impl AddressCache for Cache {
    fn get_node(&self, addr: &Address<SpineID>) -> CachedOption<NodeOffset> {
        if addr.1 != 0 {
            // We do not cache non-spine addresses atm
            return CachedOption::NoCache;
        }
        let spine_id = addr.0 .0;
        self.0
            .get(spine_id)
            .unwrap_or(&CachedOption::NoCache)
            .clone()
    }

    fn set_node(&mut self, addr: &Address<SpineID>, node: NodeOffset) {
        if addr.1 == 0 {
            let spine_id = addr.0 .0;
            if self.0.len() <= spine_id {
                self.0.resize(spine_id + 1, CachedOption::NoCache);
            }
            self.0[spine_id] = Some(node).into();
        }
    }
}

/// An Option type in the cache.
///
/// Returning [`Option::None`] would be ambiguous, as it could mean that the
/// address is not in the cache, or that the address is in the cache, but
/// the value is [`Option::None`].
#[derive(Clone, Debug)]
pub enum CachedOption<T> {
    /// The value is not in the cache.
    NoCache,
    /// The value is in the cache, and it is [`Option::None`].
    None,
    /// The value is in the cache, and it is [`Some(T)`].
    Some(T),
}

impl<T> CachedOption<T> {
    /// Get the cached value if it exists.
    ///
    /// A returned [`None`] should be taken as a definitive no, i.e.
    /// the address is in the cache AND it is [`Option::None`].
    /// This allows for short-circuiting.
    ///
    /// On the other hand, a value that is not in the cache will be returned
    /// as [`Some(None)`], meaning that the computation should go ahead
    /// and compute the value.
    pub fn cached(self) -> Option<Option<T>> {
        match self {
            CachedOption::NoCache => Some(None),
            CachedOption::None => None,
            CachedOption::Some(t) => Some(Some(t)),
        }
    }

    /// A reference to the cached value.
    pub fn as_ref(&self) -> CachedOption<&T> {
        match self {
            CachedOption::NoCache => CachedOption::NoCache,
            CachedOption::None => CachedOption::None,
            CachedOption::Some(t) => CachedOption::Some(t),
        }
    }

    /// Whether the value is not cached.
    pub fn no_cache(&self) -> bool {
        matches!(self, CachedOption::NoCache)
    }
}

impl<T> From<Option<T>> for CachedOption<T> {
    fn from(o: Option<T>) -> Self {
        match o {
            None => CachedOption::None,
            Some(t) => CachedOption::Some(t),
        }
    }
}
