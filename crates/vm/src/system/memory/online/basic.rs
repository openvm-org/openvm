use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ptr::NonNull,
};

use crate::system::memory::online::{LinearMemory, PAGE_SIZE};

pub struct BasicMemory {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl BasicMemory {
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for BasicMemory {
    fn drop(&mut self) {
        if self.size > 0 {
            // SAFETY:
            // - self.ptr is allocated via the global allocator
            // - self.layout matches the original allocation layout
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

impl Clone for BasicMemory {
    fn clone(&self) -> Self {
        if self.size == 0 {
            // Ensure we maintain the same aligned pointer for zero-size
            let aligned_ptr = PAGE_SIZE as *mut u8;
            // SAFETY:
            // - aligned_ptr is PAGE_SIZE which is non-null and properly aligned
            let ptr = unsafe { NonNull::new_unchecked(aligned_ptr) };
            return Self {
                ptr,
                size: 0,
                layout: self.layout,
            };
        }

        let layout = self.layout;
        // SAFETY:
        // - alloc_zeroed creates a valid allocation for the layout
        // - copy_nonoverlapping copies exactly self.size bytes from valid source to valid dest
        // - new_ptr is guaranteed non-null after alloc check
        let ptr = unsafe {
            let new_ptr = alloc_zeroed(layout);
            if new_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.size);
            NonNull::new_unchecked(new_ptr)
        };
        Self {
            ptr,
            size: self.size,
            layout,
        }
    }
}

impl std::fmt::Debug for BasicMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicMemory")
            .field("size", &self.size)
            .field("alignment", &self.layout.align())
            .finish()
    }
}

impl LinearMemory for BasicMemory {
    fn new(size: usize) -> Self {
        if size == 0 {
            // For zero-size allocation, use a dangling pointer with proper alignment
            // We need to ensure the pointer is aligned to PAGE_SIZE
            let aligned_ptr = PAGE_SIZE as *mut u8;
            // SAFETY:
            // - aligned_ptr is PAGE_SIZE which is non-null and properly aligned
            let ptr = unsafe { NonNull::new_unchecked(aligned_ptr) };
            let layout = Layout::from_size_align(0, PAGE_SIZE)
                .expect("Failed to create layout with PAGE_SIZE alignment");
            return Self {
                ptr,
                size: 0,
                layout,
            };
        }

        // Use PAGE_SIZE alignment for consistency with MmapMemory
        // This also ensures good alignment for any type we might store
        let layout = Layout::from_size_align(size, PAGE_SIZE)
            .expect("Failed to create layout with PAGE_SIZE alignment");

        // SAFETY:
        // - alloc_zeroed creates a valid allocation for the layout
        // - raw_ptr is guaranteed non-null after alloc check
        let ptr = unsafe {
            let raw_ptr = alloc_zeroed(layout);
            if raw_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(raw_ptr)
        };

        Self { ptr, size, layout }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY:
        // - self.ptr is valid for reads of self.size bytes
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY:
        // - self.ptr is valid for reads and writes of self.size bytes
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    #[inline(always)]
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        let size = std::mem::size_of::<BLOCK>();
        assert!(
            from + size <= self.size,
            "read from={from} of size={size} out of bounds: memory size={}",
            self.size
        );

        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `src` is aligned to `BLOCK`
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read(src)
    }

    #[inline(always)]
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        let size = std::mem::size_of::<BLOCK>();
        assert!(
            from + size <= self.size,
            "read_unaligned from={from} of size={size} out of bounds: memory size={}",
            self.size
        );

        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read_unaligned(src)
    }

    #[inline(always)]
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        let size = std::mem::size_of::<BLOCK>();
        assert!(
            start + size <= self.size,
            "write start={start} of size={size} out of bounds: memory size={}",
            self.size
        );

        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `dst` is aligned to `BLOCK`
        core::ptr::write(dst, values);
    }

    #[inline(always)]
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        let size = std::mem::size_of::<BLOCK>();
        assert!(
            start + size <= self.size,
            "write_unaligned start={start} of size={size} out of bounds: memory size={}",
            self.size
        );

        // Use slice's copy_from_slice for safe byte-level copy
        let src_bytes = std::slice::from_raw_parts(&values as *const BLOCK as *const u8, size);
        self.as_mut_slice()[start..start + size].copy_from_slice(src_bytes);
    }

    #[inline(always)]
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK) {
        let size = std::mem::size_of::<BLOCK>();
        assert!(
            start + size <= self.size,
            "swap start={start} of size={size} out of bounds: memory size={}",
            self.size
        );

        // SAFETY:
        // - Bounds check is done via assert above
        // - We assume `start` is aligned to `BLOCK`
        core::ptr::swap(
            self.as_mut_ptr().add(start) as *mut BLOCK,
            values as *mut BLOCK,
        );
    }

    #[inline(always)]
    unsafe fn copy_nonoverlapping<T: Copy>(&mut self, to: usize, data: &[T]) {
        let byte_len = std::mem::size_of_val(data);
        assert!(
            to + byte_len <= self.size,
            "copy_nonoverlapping to={to} of size={byte_len} out of bounds: memory size={}",
            self.size
        );

        // Use slice's copy_from_slice for safe byte-level copy
        let src_bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len);
        self.as_mut_slice()[to..to + byte_len].copy_from_slice(src_bytes);
    }

    #[inline(always)]
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T] {
        let byte_len = len * std::mem::size_of::<T>();
        assert!(
            start + byte_len <= self.size,
            "get_aligned_slice start={start} of size={byte_len} out of bounds: memory size={}",
            self.size
        );
        assert!(
            start % std::mem::align_of::<T>() == 0,
            "get_aligned_slice: misaligned start"
        );

        let data = self.as_ptr().add(start) as *const T;
        // SAFETY:
        // - Bounds check is done via assert above
        // - Alignment check is done via assert above
        // - `T` is "plain old data" (POD), so conversion from underlying bytes is properly
        //   initialized
        core::slice::from_raw_parts(data, len)
    }
}

// SAFETY: BasicMemory properly manages its allocation and can be sent between threads
unsafe impl Send for BasicMemory {}
// SAFETY: BasicMemory has no interior mutability and can be shared between threads
unsafe impl Sync for BasicMemory {}
