use std::hash::Hash;

use crate::{
    constraint::{self, ScopeConstraint},
    PatternID, PatternMatch,
};

pub(crate) type Constraint<P> = <P as Pattern>::Constraint;
pub(crate) type Symbol<P> = <Constraint<P> as ScopeConstraint>::Symbol;
pub(crate) type DataRef<'d, P> = <Constraint<P> as ScopeConstraint>::DataRef<'d>;
pub(crate) type Value<P> = <Constraint<P> as ScopeConstraint>::Value;
pub(crate) type ScopeMap<P> = constraint::ScopeMap<Constraint<P>>;
pub(crate) type Match<P> = PatternMatch<PatternID, Value<P>>;

/// A pattern, expressed using symbols and constraints.
///
/// Every node to be matched (the ``Universe'') is associated with a symbol,
/// and the structure
/// of the pattern is expressed as constraints on those symbols.
pub trait Pattern {
    type Constraint: ScopeConstraint + Clone;
    type Universe: Eq + Hash;

    fn constraints(&self) -> impl Iterator<Item = Self::Constraint> + '_;

    fn id(&self) -> PatternID;

    fn get_symbol(
        &self,
        u: Self::Universe,
    ) -> Option<<Self::Constraint as ScopeConstraint>::Symbol>;

    fn get_id(&self, s: <Self::Constraint as ScopeConstraint>::Symbol) -> Option<Self::Universe>;
}
