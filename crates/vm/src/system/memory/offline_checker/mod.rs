mod bridge;
mod bus;
mod columns;

pub use bridge::*;
pub use bus::*;
pub use columns::*;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

macro_rules! define_record_type {
    ($name:ident, $inner_type:ty, $size:expr) => {
        #[repr(transparent)]
        #[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Debug, Unaligned, Copy, Clone)]
        pub struct $name(pub [u8; $size]);

        impl Into<$inner_type> for $name {
            #[inline(always)]
            fn into(self) -> $inner_type {
                zerocopy::transmute!(self)
            }
        }

        impl From<$inner_type> for $name {
            #[inline(always)]
            fn from(value: $inner_type) -> Self {
                zerocopy::transmute!(value)
            }
        }

        impl $name {
            #[inline(always)]
            pub fn as_inner(&self) -> $inner_type {
                (*self).into()
            }
        }

        impl AsMut<$inner_type> for $name {
            // **SAFETY** Self must be properly aligned for the inner type
            #[inline(always)]
            fn as_mut(&mut self) -> &mut $inner_type {
                unsafe { &mut *(self.0.as_ptr() as *mut $inner_type) }
            }
        }

        impl AsRef<$inner_type> for $name {
            // **SAFETY** Self must be properly aligned for the inner type
            #[inline(always)]
            fn as_ref(&self) -> &$inner_type {
                unsafe { &*(self.0.as_ptr() as *const $inner_type) }
            }
        }
    };
}

define_record_type!(Ru32, u32, 4);
define_record_type!(Ru16, u16, 2);

#[repr(transparent)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Debug)]
pub struct MemoryReadAuxRecord {
    pub prev_timestamp: Ru32,
}

/// **SAFETY** NUM_LIMBS must be divisible by 4 so that the `prev_data` field is aligned
#[repr(C)]
#[derive(FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned, Debug)]
pub struct MemoryWriteAuxRecord<const NUM_LIMBS: usize> {
    pub prev_timestamp: Ru32,
    pub prev_data: [u8; NUM_LIMBS],
}
