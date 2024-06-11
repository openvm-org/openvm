use alloy_primitives::{U256, U512};
use std::hash::Hash;

// Note: the Data trait will likely change in the future to include more methods for accessing
// different sections of the underlying data in an more expressive way.
pub trait Data: Sized + Clone {
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

impl Index for u8 {}
impl Index for u16 {}
impl Index for u32 {}
impl Index for u64 {}
impl Index for u128 {}
impl Index for U256 {}
impl Index for U512 {}
