use crate::utils::MemorySize;
use alloy_primitives::{U256, U512};
use itertools::Itertools;
use std::hash::Hash;

// Note: the Data trait will likely change in the future to include more methods for accessing
// different sections of the underlying data in an more expressive way.
pub trait Data: Sized + Clone + MemorySize {
    fn to_be_bytes(&self) -> Vec<u8>;
    fn from_be_bytes(bytes: &[u8]) -> Option<Self>;
}

pub trait Index: Data + Hash + Eq + PartialEq {}

impl Data for u8 {
    fn to_be_bytes(&self) -> Vec<u8> {
        vec![*self]
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(bytes[0])
    }
}

impl Data for u16 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }
}

impl Data for u32 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u32::from_be_bytes(bytes.try_into().ok()?))
    }
}

impl Data for u64 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u64::from_be_bytes(bytes.try_into().ok()?))
    }
}

impl Data for u128 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(u128::from_be_bytes(bytes.try_into().ok()?))
    }
}

impl Data for U256 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes::<32>().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(U256::from_be_bytes::<32>(bytes.try_into().ok()?))
    }
}

impl Data for U512 {
    fn to_be_bytes(&self) -> Vec<u8> {
        (*self).to_be_bytes::<64>().to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        Some(U512::from_be_bytes::<64>(bytes.try_into().ok()?))
    }
}

impl<const N: usize> Data for [u8; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.to_vec()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes.try_into().ok()
    }
}

impl<const N: usize> Data for [u16; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter().rev().flat_map(|x| x.to_be_bytes()).collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(2)
            .map(|x| u16::from_be_bytes(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl<const N: usize> Data for [u32; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter().rev().flat_map(|x| x.to_be_bytes()).collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(4)
            .map(|x| u32::from_be_bytes(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl<const N: usize> Data for [u64; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter().rev().flat_map(|x| x.to_be_bytes()).collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(8)
            .map(|x| u64::from_be_bytes(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl<const N: usize> Data for [u128; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter().rev().flat_map(|x| x.to_be_bytes()).collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(16)
            .map(|x| u128::from_be_bytes(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl<const N: usize> Data for [U256; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter()
            .rev()
            .flat_map(|x| x.to_be_bytes::<32>())
            .collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(32)
            .map(|x| U256::from_be_bytes::<32>(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl<const N: usize> Data for [U512; N] {
    fn to_be_bytes(&self) -> Vec<u8> {
        self.iter()
            .rev()
            .flat_map(|x| x.to_be_bytes::<64>())
            .collect()
    }

    fn from_be_bytes(bytes: &[u8]) -> Option<Self> {
        bytes
            .chunks(64)
            .map(|x| U512::from_be_bytes::<64>(x.try_into().unwrap()))
            .collect_vec()
            .try_into()
            .ok()
    }
}

impl Index for u8 {}
impl Index for u16 {}
impl Index for u32 {}
impl Index for u64 {}
impl Index for u128 {}
impl Index for U256 {}
impl Index for U512 {}
impl<const N: usize> Index for [u8; N] {}
impl<const N: usize> Index for [u16; N] {}
impl<const N: usize> Index for [u32; N] {}
impl<const N: usize> Index for [u64; N] {}
impl<const N: usize> Index for [u128; N] {}
impl<const N: usize> Index for [U256; N] {}
impl<const N: usize> Index for [U512; N] {}
