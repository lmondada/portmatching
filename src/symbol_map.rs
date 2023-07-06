//! A map from symbols to Us that is fast
//! 

use std::collections::BTreeSet;

use crate::{automaton::StateID, predicate::Symbol};

pub(crate) struct SymbolMap<U> {
    data: Vec<Option<U>>,
    data_set: BTreeSet<U>,
    state_id: StateID
}

impl<U: Ord> SymbolMap<U> {
    pub(crate) fn new(root_state: StateID, root: U) -> Self {
        let mut s = Self {
            data: vec![],
            data_set: BTreeSet::new(),
            state_id: root_state
        };
        s.insert(Symbol::root(), root);
        s
    }

    pub(crate) fn insert(&mut self, symb: Symbol, value: U) {
        let ind = Self::get_ind(symb);
        if ind > self.data.len() {
            self.data.resize(ind + 1);
        }
        self.data[ind] = Some(value);
    }

    fn get_ind(symb: Symbol) -> usize {
        let Symbol(status, ind) = symb;

    }
}