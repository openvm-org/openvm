use openvm_circuit_primitives_derive::AlignedBorrow;

pub mod adapter;
mod controller;
// #[cfg(any(unix, windows))]
// pub mod memmap;
pub mod merkle;
pub mod offline_checker;
pub mod online;
// #[cfg(not(any(unix, windows)))]
pub mod paged_vec;
mod persistent;
// TODO: add back
// #[cfg(test)]
// mod tests;
pub mod tree;
mod volatile;

pub use controller::*;
// #[cfg(any(unix, windows))]
// pub use memmap::*;
pub use online::INITIAL_TIMESTAMP;
// #[cfg(not(any(unix, windows)))]
pub use paged_vec::*;

// #[cfg(any(unix, windows))]
// pub type MemoryBackend = MmapWrapper;
// #[cfg(not(any(unix, windows)))]
pub type MemoryBackend = PagedVec;

#[derive(PartialEq, Copy, Clone, Debug, Eq)]
pub enum OpType {
    Read = 0,
    Write = 1,
}

/// The full pointer to a location in memory consists of an address space and a pointer within
/// the address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
#[repr(C)]
pub struct MemoryAddress<S, T> {
    pub address_space: S,
    pub pointer: T,
}

impl<S, T> MemoryAddress<S, T> {
    pub fn new(address_space: S, pointer: T) -> Self {
        Self {
            address_space,
            pointer,
        }
    }

    pub fn from<T1, T2>(a: MemoryAddress<T1, T2>) -> Self
    where
        T1: Into<S>,
        T2: Into<T>,
    {
        Self {
            address_space: a.address_space.into(),
            pointer: a.pointer.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, AlignedBorrow)]
#[repr(C)]
pub struct HeapAddress<S, T> {
    pub address: MemoryAddress<S, T>,
    pub data: MemoryAddress<S, T>,
}
