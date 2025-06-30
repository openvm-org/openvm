use derive_new::new;

/// A struct that has the same memory layout as `uint2` to be used in FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, new)]
pub struct UInt2 {
    pub x: u32,
    pub y: u32,
}
