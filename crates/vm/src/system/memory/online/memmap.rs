use std::fmt::Debug;

use memmap2::MmapMut;

use super::{LinearMemory, PAGE_SIZE};

pub const CELL_STRIDE: usize = 1;

/// Mmap-backed linear memory. OS-memory pages are paged in on-demand and zero-initialized.
#[derive(Debug)]
pub struct MmapMemory {
    mmap: MmapMut,
}

impl Clone for MmapMemory {
    fn clone(&self) -> Self {
        let mut new_mmap = MmapMut::map_anon(self.mmap.len()).unwrap();
        new_mmap.copy_from_slice(&self.mmap);
        Self { mmap: new_mmap }
    }
}

impl MmapMemory {
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }
}

impl LinearMemory for MmapMemory {
    /// Create a new MmapMemory with the given `size` in bytes.
    /// We round `size` up to be a multiple of the mmap page size (4kb by default) so that OS-level
    /// MMU protection corresponds to out of bounds protection.
    fn new(mut size: usize) -> Self {
        size = size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        // anonymous mapping means pages are zero-initialized on first use
        Self {
            mmap: MmapMut::map_anon(size).unwrap(),
        }
    }

    fn size(&self) -> usize {
        self.mmap.len()
    }

    fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.mmap
    }

    #[inline(always)]
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        debug_assert!(
            from + size_of::<BLOCK>() <= self.size(),
            "read from={from} of size={} out of bounds: memory size={}",
            size_of::<BLOCK>(),
            self.size()
        );
        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - MMU will segfault if `src` access is out of bounds.
        // - We assume `src` is aligned to `BLOCK`
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read(src)
    }

    #[inline(always)]
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        debug_assert!(
            from + size_of::<BLOCK>() <= self.size(),
            "read_unaligned from={from} of size={} out of bounds: memory size={}",
            size_of::<BLOCK>(),
            self.size()
        );
        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - MMU will segfault if `src` access is out of bounds.
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read_unaligned(src)
    }

    #[inline(always)]
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        debug_assert!(
            start + size_of::<BLOCK>() <= self.size(),
            "write start={start} of size={} out of bounds: memory size={}",
            size_of::<BLOCK>(),
            self.size()
        );
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - MMU will segfault if `dst` access is out of bounds.
        // - We assume `dst` is aligned to `BLOCK`
        core::ptr::write(dst, values);
    }

    #[inline(always)]
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        debug_assert!(
            start + size_of::<BLOCK>() <= self.size(),
            "write_unaligned start={start} of size={} out of bounds: memory size={}",
            size_of::<BLOCK>(),
            self.size()
        );
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - MMU will segfault if `dst` access is out of bounds.
        core::ptr::write_unaligned(dst, values);
    }

    #[inline(always)]
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK) {
        debug_assert!(
            start + size_of::<BLOCK>() <= self.size(),
            "swap start={start} of size={} out of bounds: memory size={}",
            size_of::<BLOCK>(),
            self.size()
        );
        // SAFETY:
        // - MMU will segfault if `start` access is out of bounds.
        // - We assume `start` is aligned to `BLOCK`
        core::ptr::swap(
            self.as_mut_ptr().add(start) as *mut BLOCK,
            values as *mut BLOCK,
        );
    }

    #[inline(always)]
    unsafe fn copy_nonoverlapping<T: Copy>(&mut self, to: usize, data: &[T]) {
        debug_assert!(
            to + size_of_val(data) <= self.size(),
            "copy_nonoverlapping to={to} of size={} out of bounds: memory size={}",
            size_of_val(data),
            self.size()
        );
        debug_assert_eq!(PAGE_SIZE % align_of::<T>(), 0);
        let src = data.as_ptr();
        let dst = self.as_mut_ptr().add(to) as *mut T;
        // SAFETY:
        // - MMU will segfault if `dst..dst + size_of_val(data)` is out of bounds.
        // - Assumes `to` is aligned to `T` and `self.as_mut_ptr()` is aligned to `T`, which implies
        //   the same for `dst`.
        core::ptr::copy_nonoverlapping::<T>(src, dst, data.len());
    }

    #[inline(always)]
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T] {
        debug_assert!(
            start + len * size_of::<T>() <= self.size(),
            "get_aligned_slice start={start} of size={} out of bounds: memory size={}",
            len * size_of::<T>(),
            self.size()
        );
        let data = self.as_ptr().add(start) as *const T;
        // SAFETY:
        // - MMU will segfault if `data..data + len * size_of::<T>()` is out of bounds.
        // - Assumes `data` is aligned to `T`
        // - `T` is "plain old data" (POD), so conversion from underlying bytes is properly
        //   initialized
        // - `self` will not be mutated while borrowed
        core::slice::from_raw_parts(data, len)
    }
}
