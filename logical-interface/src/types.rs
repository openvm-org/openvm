use std::hash::Hash;

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

impl Index for u8 {}
impl Index for u16 {}
impl Index for u32 {}
impl Index for u64 {}
