extern crate alloc;
use {core::mem::MaybeUninit};

use alloc::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use core::ptr::NonNull;

// Register, Memory, and IO address spaces require 4-byte alignment
pub const MIN_ALIGN: usize = 4;

/// Bytes aligned to 4 bytes.
pub struct AlignedBuf {
    pub ptr: *mut u8,
    pub layout: Layout,
}

impl AlignedBuf {
    /// Allocate a new buffer whose start address is aligned to 4 bytes.
    pub fn uninit(len: usize) -> Self {
        let layout = Layout::from_size_align(len, MIN_ALIGN).unwrap();
        if layout.size() == 0 {
            return Self {
                ptr: NonNull::<u32>::dangling().as_ptr() as *mut u8,
                layout,
            };
        }
        // SAFETY: `len` is nonzero
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        AlignedBuf { ptr, layout }
    }

    /// Allocate a new buffer whose start address is aligned to 4 bytes
    /// and copy the given data into it.
    ///
    /// # Safety
    /// - `bytes` must not be null
    /// - `len` should not be zero
    ///
    /// See [alloc]. In particular `data` should not be empty.
    pub unsafe fn new(bytes: *const u8, len: usize) -> Self {
        let buf = Self::uninit(len);
        // SAFETY:
        // - src and dst are not null
        // - src and dst are allocated for size
        // - no alignment requirements on u8
        // - non-overlapping since ptr is newly allocated
        unsafe {
            core::ptr::copy_nonoverlapping(bytes, buf.ptr, len);
        }

        buf
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}
