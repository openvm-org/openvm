use afs_derive::AlignedBorrow;

mod adapter;
mod manager;
pub mod merkle;
pub mod offline_checker;
mod persistent;
#[cfg(test)]
mod tests;
pub mod tree;
mod volatile;

pub use manager::*;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// The full identifier of a location in memory consists of an address space and a address within
/// the address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
#[repr(C)]
pub struct MemoryAddress<S, T> {
    pub address_space: S,
    pub address: T,
}

impl<S, T> MemoryAddress<S, T> {
    pub fn new(address_space: S, address: T) -> Self {
        Self {
            address_space,
            address,
        }
    }

    pub fn from<T1, T2>(a: MemoryAddress<T1, T2>) -> Self
    where
        T1: Into<S>,
        T2: Into<T>,
    {
        Self {
            address_space: a.address_space.into(),
            address: a.address.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
#[repr(C)]
pub struct HeapAddress<S, T> {
    pub address: MemoryAddress<S, T>,
    pub data: MemoryAddress<S, T>,
}
