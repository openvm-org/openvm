use std::{
    ffi::c_void,
    fmt::Debug,
    mem::{align_of, size_of, size_of_val},
    num::NonZeroUsize,
    ptr::NonNull,
    slice,
    sync::OnceLock,
};

#[cfg(target_os = "linux")]
use libc::{madvise, MADV_DONTNEED};
use nix::sys::mman::{mmap_anonymous, mprotect, munmap, MapFlags, ProtFlags};

use super::{LinearMemory, MmapMemory, PAGE_SIZE};

/// Minimum size of each inaccessible region surrounding guarded memory.
///
/// The actual guard size is rounded up to a multiple of the host OS page size.
const MIN_GUARD_SIZE: usize = 1 << 14;

static HOST_PAGE_SIZE: OnceLock<usize> = OnceLock::new();

enum MemoryAllocation {
    /// Exact `[PROT_NONE guard][RW memory][PROT_NONE guard]` mapping.
    Guarded {
        region: NonNull<c_void>,
        total_size: usize,
        guard_size: usize,
    },
    /// Logical sizes that are not OS-page-aligned cannot have exact guards on both sides.
    Plain(MmapMemory),
}

/// Mmap-backed linear memory that uses exact guard regions when its logical size is
/// OS-page-aligned.
///
/// RVR's main memory is a large power-of-two allocation, so its layout is exactly
/// `[PROT_NONE guard][RW memory][PROT_NONE guard]`. An access that crosses either adjacent
/// boundary faults at the OS level. Guard regions are defense-in-depth for RVR's raw native memory
/// accesses; software bounds checks remain authoritative when enabled.
pub struct GuardedMemory {
    allocation: MemoryAllocation,
    /// Logical memory size in bytes.
    memory_size: usize,
}

impl GuardedMemory {
    fn host_page_size() -> usize {
        *HOST_PAGE_SIZE.get_or_init(|| {
            let page_size = unsafe { nix::libc::sysconf(nix::libc::_SC_PAGESIZE) };
            let page_size = usize::try_from(page_size)
                .expect("sysconf(_SC_PAGESIZE) must return a positive value");
            assert!(page_size > 0, "sysconf(_SC_PAGESIZE) returned zero");
            page_size
        })
    }

    fn guard_size(page_size: usize) -> usize {
        MIN_GUARD_SIZE.div_ceil(page_size) * page_size
    }

    /// Whether this allocation has exact inaccessible regions on both sides.
    #[inline(always)]
    pub fn is_guarded(&self) -> bool {
        matches!(self.allocation, MemoryAllocation::Guarded { .. })
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        match &self.allocation {
            MemoryAllocation::Guarded {
                region, guard_size, ..
            } => unsafe { region.as_ptr().cast::<u8>().add(*guard_size) },
            MemoryAllocation::Plain(memory) => memory.as_ptr(),
        }
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.allocation {
            MemoryAllocation::Guarded {
                region, guard_size, ..
            } => unsafe { region.as_ptr().cast::<u8>().add(*guard_size) },
            MemoryAllocation::Plain(memory) => memory.as_mut_ptr(),
        }
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
        match &self.allocation {
            MemoryAllocation::Guarded { .. } => {
                let mut cloned = Self::new(self.memory_size);
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        self.as_ptr(),
                        cloned.as_mut_ptr(),
                        self.memory_size,
                    );
                }
                cloned
            }
            MemoryAllocation::Plain(memory) => Self {
                allocation: MemoryAllocation::Plain(memory.clone()),
                memory_size: self.memory_size,
            },
        }
    }
}

impl Debug for GuardedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedMemory")
            .field("memory_size", &self.memory_size)
            .field("is_guarded", &self.is_guarded())
            .finish()
    }
}

impl Drop for GuardedMemory {
    fn drop(&mut self) {
        if let MemoryAllocation::Guarded {
            region, total_size, ..
        } = &self.allocation
        {
            unsafe {
                let _ = munmap(*region, *total_size);
            }
        }
    }
}

impl LinearMemory for GuardedMemory {
    fn new(size: usize) -> Self {
        let page_size = Self::host_page_size();
        if size == 0 || !size.is_multiple_of(page_size) {
            return Self {
                allocation: MemoryAllocation::Plain(MmapMemory::new(size)),
                memory_size: size,
            };
        }

        let guard_size = Self::guard_size(page_size);
        let total_size = size
            .checked_add(2 * guard_size)
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

        let usable_start = unsafe {
            NonNull::new_unchecked(
                region
                    .as_ptr()
                    .cast::<u8>()
                    .add(guard_size)
                    .cast::<c_void>(),
            )
        };
        if let Err(error) = unsafe {
            mprotect(
                usable_start,
                size,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            )
        } {
            unsafe {
                let _ = munmap(region, total_size);
            }
            panic!("mprotect failed for guarded memory: {error}");
        }

        Self {
            allocation: MemoryAllocation::Guarded {
                region,
                total_size,
                guard_size,
            },
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

    #[cfg(target_os = "linux")]
    fn fill_zero(&mut self) {
        if let MemoryAllocation::Plain(memory) = &mut self.allocation {
            memory.fill_zero();
            return;
        }

        // SAFETY: the usable region is an anonymous private mapping (PROT_READ|PROT_WRITE).
        // MADV_DONTNEED on anonymous private mappings causes subsequent accesses to return
        // zero-filled pages without writing anything. We advise only [as_mut_ptr(), +memory_size),
        // leaving the PROT_NONE guard pages untouched.
        unsafe {
            let ret = madvise(
                self.as_mut_ptr() as *mut libc::c_void,
                self.memory_size,
                MADV_DONTNEED,
            );
            if ret != 0 {
                std::ptr::write_bytes(self.as_mut_ptr(), 0, self.memory_size);
            }
        }
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

#[cfg(not(feature = "unprotected"))]
#[cold]
#[inline(never)]
fn panic_oob(start: usize, size: usize, memory_size: usize) -> ! {
    panic!("Memory access out of bounds: start={start} size={size} memory_size={memory_size}");
}

#[cfg(test)]
mod tests {
    use openvm_platform::memory::MEM_SIZE;

    use super::*;

    #[test]
    fn only_page_aligned_memory_uses_exact_guards() {
        let page_size = GuardedMemory::host_page_size();
        assert!(GuardedMemory::new(page_size).is_guarded());
        assert!(!GuardedMemory::new(page_size - 1).is_guarded());
        assert!(!GuardedMemory::new(0).is_guarded());

        let rvr_memory = GuardedMemory::new(MEM_SIZE);
        assert!(rvr_memory.is_guarded());
    }

    #[test]
    fn guarded_boundaries_fault() {
        let page_size = GuardedMemory::host_page_size();
        let mut memory = GuardedMemory::new(2 * page_size);
        assert!(memory.is_guarded());

        memory.as_mut_slice().fill(0xff);
        memory.fill_zero();
        assert!(memory.as_slice().iter().all(|&byte| byte == 0));

        unsafe {
            memory.as_mut_ptr().write(1);
            memory.as_mut_ptr().add(memory.size() - 1).write(2);
            assert_access_faults(memory.as_ptr().sub(1));
            assert_access_faults(memory.as_ptr().add(memory.size()));
        }
    }

    fn assert_access_faults(ptr: *const u8) {
        let pid = unsafe { nix::libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            unsafe {
                std::ptr::read_volatile(ptr);
                nix::libc::_exit(0);
            }
        }

        let mut status = 0;
        assert_eq!(unsafe { nix::libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(nix::libc::WIFSIGNALED(status));
        let signal = nix::libc::WTERMSIG(status);
        assert!(
            signal == nix::libc::SIGSEGV || signal == nix::libc::SIGBUS,
            "guard access terminated with unexpected signal {signal}"
        );
    }
}
