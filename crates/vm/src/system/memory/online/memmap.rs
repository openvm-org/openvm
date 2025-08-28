use std::{
    fmt::Debug,
    mem::{align_of, size_of, size_of_val},
};

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

    #[cfg(not(feature = "unprotected"))]
    #[inline(always)]
    fn check_bounds(&self, start: usize, size: usize) {
        let memory_size = self.size();
        if start > memory_size || size > memory_size - start {
            panic_oob(start, size, memory_size);
        }
    }

    #[cfg(feature = "unprotected")]
    #[inline(always)]
    fn check_bounds(&self, start: usize, size: usize) {
        let memory_size = self.size();
        debug_assert!(
            start <= memory_size && size <= memory_size - start,
            "Memory access out of bounds: start={} size={} memory_size={}",
            start,
            size,
            memory_size
        );
    }
}

impl LinearMemory for MmapMemory {
    /// Create a new MmapMemory with the given `size` in bytes.
    /// We round `size` up to be a multiple of the mmap page size (4kb by default).
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

    #[cfg(target_os = "linux")]
    fn fill_zero(&mut self) {
        use libc::{madvise, MADV_DONTNEED};

        let mmap = &mut self.mmap;
        // SAFETY: our mmap is a memory-backed (not file-backed) anonymous private mapping.
        // When we madvise MADV_DONTNEED, according to https://man7.org/linux/man-pages/man2/madvise.2.html
        // > subsequent accesses of pages in the range will succeed, but
        // > will result in either repopulating the memory contents from
        // > the up-to-date contents of the underlying mapped file (for
        // > shared file mappings, shared anonymous mappings, and shmem-
        // > based techniques such as System V shared memory segments)
        // > or zero-fill-on-demand pages for anonymous private
        // > mappings.
        unsafe {
            let ret = madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                MADV_DONTNEED,
            );
            if ret != 0 {
                // Fallback to write_bytes if madvise fails
                std::ptr::write_bytes(mmap.as_mut_ptr(), 0, mmap.len());
            }
        }
    }

    #[inline(always)]
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        self.check_bounds(from, size_of::<BLOCK>());
        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - We assume `src` is aligned to `BLOCK`
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read(src)
    }

    #[inline(always)]
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        self.check_bounds(from, size_of::<BLOCK>());
        let src = self.as_ptr().add(from) as *const BLOCK;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - We assume `BLOCK` is "plain old data" so the underlying `src` bytes is valid to read as
        //   an initialized value of `BLOCK`
        core::ptr::read_unaligned(src)
    }

    #[inline(always)]
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - We assume `dst` is aligned to `BLOCK`
        core::ptr::write(dst, values);
    }

    #[inline(always)]
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        core::ptr::write_unaligned(dst, values);
    }

    #[inline(always)]
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - We assume `start` is aligned to `BLOCK`
        core::ptr::swap(
            self.as_mut_ptr().add(start) as *mut BLOCK,
            values as *mut BLOCK,
        );
    }

    #[inline(always)]
    unsafe fn copy_nonoverlapping<T: Copy>(&mut self, to: usize, data: &[T]) {
        self.check_bounds(to, size_of_val(data));
        debug_assert_eq!(PAGE_SIZE % align_of::<T>(), 0);
        let src = data.as_ptr();
        let dst = self.as_mut_ptr().add(to) as *mut T;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - Assumes `to` is aligned to `T` and `self.as_mut_ptr()` is aligned to `T`, which implies
        //   the same for `dst`.
        core::ptr::copy_nonoverlapping::<T>(src, dst, data.len());
    }

    #[inline(always)]
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T] {
        self.check_bounds(start, len * size_of::<T>());
        let data = self.as_ptr().add(start) as *const T;
        // SAFETY:
        // - Bounds checked above (unless unprotected feature enabled)
        // - Assumes `data` is aligned to `T`
        // - `T` is "plain old data" (POD), so conversion from underlying bytes is properly
        //   initialized
        // - `self` will not be mutated while borrowed
        core::slice::from_raw_parts(data, len)
    }
}

#[cold]
#[inline(never)]
fn panic_oob(start: usize, size: usize, memory_size: usize) -> ! {
    panic!(
        "Memory access out of bounds: start={} size={} memory_size={}",
        start, size, memory_size
    );
}
