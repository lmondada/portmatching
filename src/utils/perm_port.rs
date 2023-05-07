use std::{ops::{Index, IndexMut}, mem};

use portgraph::{SecondaryMap, PortIndex};

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PermPortIndex(usize);

impl PermPortIndex {
    pub(crate) fn next(&self) -> Self {
        let PermPortIndex(ind) = self;
        PermPortIndex(ind + 1)
    }
}

impl Into<usize> for PermPortIndex {
    fn into(self) -> usize {
        let PermPortIndex(ind) = self;
        ind
    }
}

#[derive(Clone, Default, Debug)]
pub struct PermPortPool {
    to_port: SecondaryMap<PermPortIndex, Option<PortIndex>>,
    to_perm: SecondaryMap<PortIndex, Option<PermPortIndex>>,
    ind: usize,
}

impl PermPortPool {
    pub(crate) fn new() -> Self {
        PermPortPool::default()
    }

    pub(crate) fn rekey_fn(&mut self) -> impl FnMut(PortIndex, Option<PortIndex>) + '_ {
        |old, new| {
            let Some(perm) = mem::take(&mut self.to_perm[old]) else { return };
            self.to_port[perm] = Default::default();
            let Some(new) = new else { return };
            self.to_perm[new] = Some(perm);
            self.to_port[perm] = Some(new);
        }
    }

    pub(crate) fn rekey(&mut self, old: PortIndex, new: Option<PortIndex>) {
        self.rekey_fn()(old, new);
    }

    pub(crate) fn get_or_create_perm(&mut self, port: PortIndex) -> PermPortIndex {
        if let Some(perm) = self[port] {
            return perm;
        } else {
            return self.create_perm(port);
        }
    }

    pub(crate) fn create_perm(&mut self, port: PortIndex) -> PermPortIndex {
        let perm = PermPortIndex(self.ind);
        self.ind += 1;
        self.to_port[perm] = Some(port);
        self.to_perm[port] = Some(perm);
        perm
    }
}

impl Index<PermPortIndex> for PermPortPool {
    type Output = PortIndex;

    fn index(&self, index: PermPortIndex) -> &Self::Output {
        self.to_port[index].as_ref().expect("invalid index")
    }
}

impl Index<PortIndex> for PermPortPool {
    type Output = Option<PermPortIndex>;

    fn index(&self, index: PortIndex) -> &Self::Output {
        &self.to_perm[index]
    }
}