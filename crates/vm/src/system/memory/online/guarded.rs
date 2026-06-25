use std::{
    ffi::c_void,
    fmt::Debug,
    mem::{align_of, size_of, size_of_val},
    num::NonZeroUsize,
    ptr::NonNull,
    slice,
};

use nix::sys::mman::{mmap_anonymous, mprotect, munmap, MapFlags, ProtFlags};

use super::{LinearMemory, PAGE_SIZE};

pub const GUARD_SIZE: usize = 1 << 14;

pub struct GuardedMemory {
    region: NonNull<c_void>,
    /// Total mmap'd size including both guard regions.
    total_size: usize,
    /// Usable memory size (does not include guards).
    memory_size: usize,
}

impl GuardedMemory {
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.region.as_ptr().cast::<u8>().add(GUARD_SIZE) }
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.region.as_ptr().cast::<u8>().add(GUARD_SIZE) }
    }

    #[cfg(not(feature = "unprotected"))]
    #[inline(always)]
    fn check_bounds(&self, start: usize, size: usize) {
        if start > self.memory_size || size > self.memory_size - start {
            panic_oob(start, size, self.memory_size);
        }
    }

    #[cfg(feature = "unprotected")]
    #[inline(always)]
    fn check_bounds(&self, start: usize, size: usize) {
        debug_assert!(
            start <= self.memory_size && size <= self.memory_size - start,
            "Memory access out of bounds: start={} size={} memory_size={}",
            start,
            size,
            self.memory_size
        );
    }
}

impl Clone for GuardedMemory {
    fn clone(&self) -> Self {
        let mut cloned = Self::new(self.memory_size);
        if self.memory_size > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.as_ptr(),
                    cloned.as_mut_ptr(),
                    self.memory_size,
                );
            }
        }
        cloned
    }
}

impl Debug for GuardedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedMemory")
            .field("total_size", &self.total_size)
            .field("memory_size", &self.memory_size)
            .finish()
    }
}

impl Drop for GuardedMemory {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.region, self.total_size);
        }
    }
}

impl LinearMemory for GuardedMemory {
    fn new(size: usize) -> Self {
        let total_size = size
            .checked_add(2 * GUARD_SIZE)
            .expect("guarded memory size overflow");
        let total_size_nz =
            NonZeroUsize::new(total_size).expect("guarded memory size must be nonzero");

        let region = unsafe {
            mmap_anonymous(
                None,
                total_size_nz,
                ProtFlags::PROT_NONE,
                MapFlags::MAP_PRIVATE | MapFlags::MAP_NORESERVE,
            )
            .unwrap()
        };

        if size > 0 {
            let usable_start = unsafe {
                NonNull::new_unchecked(
                    region
                        .as_ptr()
                        .cast::<u8>()
                        .add(GUARD_SIZE)
                        .cast::<c_void>(),
                )
            };
            unsafe {
                mprotect(usable_start, size, ProtFlags::PROT_READ | ProtFlags::PROT_WRITE)
                    .unwrap();
            }
        }

        Self {
            region,
            total_size,
            memory_size: size,
        }
    }

    fn size(&self) -> usize {
        self.memory_size
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.memory_size) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.memory_size) }
    }

    #[inline(always)]
    unsafe fn read<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        self.check_bounds(from, size_of::<BLOCK>());
        let src = self.as_ptr().add(from) as *const BLOCK;
        core::ptr::read(src)
    }

    #[inline(always)]
    unsafe fn read_unaligned<BLOCK: Copy>(&self, from: usize) -> BLOCK {
        self.check_bounds(from, size_of::<BLOCK>());
        let src = self.as_ptr().add(from) as *const BLOCK;
        core::ptr::read_unaligned(src)
    }

    #[inline(always)]
    unsafe fn write<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        core::ptr::write(dst, values);
    }

    #[inline(always)]
    unsafe fn write_unaligned<BLOCK: Copy>(&mut self, start: usize, values: BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
        let dst = self.as_mut_ptr().add(start) as *mut BLOCK;
        core::ptr::write_unaligned(dst, values);
    }

    #[inline(always)]
    unsafe fn swap<BLOCK: Copy>(&mut self, start: usize, values: &mut BLOCK) {
        self.check_bounds(start, size_of::<BLOCK>());
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
        core::ptr::copy_nonoverlapping::<T>(src, dst, data.len());
    }

    #[inline(always)]
    unsafe fn get_aligned_slice<T: Copy>(&self, start: usize, len: usize) -> &[T] {
        self.check_bounds(start, len * size_of::<T>());
        let data = self.as_ptr().add(start) as *const T;
        slice::from_raw_parts(data, len)
    }
}

unsafe impl Send for GuardedMemory {}
unsafe impl Sync for GuardedMemory {}

#[cold]
#[inline(never)]
fn panic_oob(start: usize, size: usize, memory_size: usize) -> ! {
    panic!("Memory access out of bounds: start={start} size={size} memory_size={memory_size}");
}
