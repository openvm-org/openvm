mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

#[repr(transparent)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Unaligned, Copy, Clone)]
pub struct Ru32(pub [u8; 4]);

impl Into<u32> for Ru32 {
    #[inline(always)]
    fn into(self) -> u32 {
        zerocopy::transmute!(self)
    }
}

impl From<u32> for Ru32 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        zerocopy::transmute!(value)
    }
}

impl Ru32 {
    #[inline(always)]
    pub fn as_u32(&self) -> u32 {
        (*self).into()
    }
}

/// **SAFETY** Self must be 4-byte aligned
impl<'a> Into<&'a u32> for &'a Ru32 {
    #[inline(always)]
    fn into(self) -> &'a u32 {
        unsafe { &*(self.0.as_ptr() as *const u32) }
    }
}

/// **SAFETY** Self must be 4-byte aligned
impl<'a> Into<&'a mut u32> for &'a mut Ru32 {
    #[inline(always)]
    fn into(self) -> &'a mut u32 {
        unsafe { &mut *(self.0.as_ptr() as *mut u32) }
    }
}

#[repr(transparent)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Unaligned)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: Ru32,
}

/// **SAFETY** NUM_LIMBS must be divisible by 4 so that the `prev_data` field is aligned
#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct MemoryWriteAuxRecord<const NUM_LIMBS: usize> {
    pub prev_timestamp: Ru32,
    pub prev_data: [u8; NUM_LIMBS],
}
