use core::alloc::{GlobalAlloc, Layout};

use crate::memory::sys_alloc_aligned;

#[global_allocator]
pub static HEAP: BumpPointerAlloc = BumpPointerAlloc;

pub struct BumpPointerAlloc;

unsafe impl GlobalAlloc for BumpPointerAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        sys_alloc_aligned(layout.size(), layout.align())
    }

    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
        // this allocator never deallocates memory
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // NOTE: This is safe to avoid zeroing allocated bytes, as the bump allocator does not
        //       reuse memory and the zkVM memory is zero-initialized.
        self.alloc(layout)
    }
}
