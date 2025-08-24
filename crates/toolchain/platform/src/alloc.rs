extern crate alloc;

use alloc::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use core::ptr::NonNull;

/// Bytes allocated according to the given Layout
pub struct AlignedBuf {
    pub ptr: *mut u8,
    pub layout: Layout,
}

impl AlignedBuf {
    /// Allocate a new buffer whose start address is aligned to `align` bytes.
    /// *NOTE* if `len` is zero then a creates new `NonNull` that is dangling and 16-byte aligned.
    pub fn uninit(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len, align).unwrap();
        if layout.size() == 0 {
            return Self {
                ptr: NonNull::<u128>::dangling().as_ptr() as *mut u8,
                layout,
            };
        }
        // TODO[jpw]: replace `alloc` with `allocate` once the `Allocator` trait is stabilized.
        //            (see https://doc.rust-lang.org/alloc/alloc/fn.alloc.html)
        // SAFETY: `len` is nonzero
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        AlignedBuf { ptr, layout }
    }

    /// Allocate a new buffer whose start address is aligned to `align` bytes
    /// and copy the given data into it.
    ///
    /// # Safety
    /// - `bytes` must not be null
    /// - `len` should not be zero
    ///
    /// See [alloc]. In particular `data` should not be empty.
    pub unsafe fn new(bytes: *const u8, len: usize, align: usize) -> Self {
        let buf = Self::uninit(len, align);
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
            // SAFETY:
            // - self.ptr was allocated with self.layout
            // - Pointer and layout are valid from creation
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}
