//! Guarded memory allocation with mmap.
//!
//! Provides a memory region with guard pages on each side to catch
//! buffer overflows/underflows at the OS level.

use std::{ffi::c_void, num::NonZeroUsize, ptr::NonNull};

use nix::sys::mman::{mmap_anonymous, mprotect, munmap, MapFlags, ProtFlags};
use thiserror::Error;

/// Guard page size (16KB, must be >= page size and cover max load/store offset).
pub const GUARD_SIZE: usize = 1 << 14;

/// Memory allocation error.
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("mmap failed: {0}")]
    MmapFailed(#[from] nix::Error),

    #[error("invalid memory size: {0}")]
    InvalidSize(usize),
}

/// Memory region with guard pages.
///
/// Allocates `[GUARD][MEMORY][GUARD]` with the guard pages protected as `PROT_NONE`.
/// Any access to guard pages will cause a segfault, catching buffer overflows.
pub struct GuardedMemory {
    /// Pointer to the start of the entire region (including first guard).
    region: NonNull<c_void>,
    /// Total size including both guard pages.
    total_size: usize,
    /// Size of the usable memory region.
    memory_size: usize,
}

impl GuardedMemory {
    /// Allocate a new guarded memory region.
    pub fn new(memory_size: usize) -> Result<Self, MemoryError> {
        if memory_size == 0 {
            return Err(MemoryError::InvalidSize(memory_size));
        }

        let total_size = memory_size
            .checked_add(2 * GUARD_SIZE)
            .ok_or(MemoryError::InvalidSize(memory_size))?;
        let total_size_nz =
            NonZeroUsize::new(total_size).ok_or(MemoryError::InvalidSize(memory_size))?;

        // Allocate entire region as PROT_NONE
        let region = unsafe {
            mmap_anonymous(
                None,
                total_size_nz,
                ProtFlags::PROT_NONE,
                MapFlags::MAP_PRIVATE | MapFlags::MAP_NORESERVE,
            )?
        };

        // Make middle portion readable/writable
        let memory_start = unsafe {
            NonNull::new_unchecked(
                region
                    .as_ptr()
                    .cast::<u8>()
                    .add(GUARD_SIZE)
                    .cast::<c_void>(),
            )
        };
        unsafe {
            mprotect(
                memory_start,
                memory_size,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
            )?;
        }

        Ok(Self {
            region,
            total_size,
            memory_size,
        })
    }

    /// Returns pointer to usable memory (after first guard page).
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        unsafe { self.region.as_ptr().cast::<u8>().add(GUARD_SIZE) }
    }

    /// Returns the size of the usable memory region.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.memory_size
    }

    /// Copy data into memory at the given offset.
    ///
    /// # Safety
    ///
    /// Caller must ensure `offset + data.len() <= self.size()`.
    pub unsafe fn copy_from(&mut self, offset: usize, data: &[u8]) {
        debug_assert!(offset + data.len() <= self.memory_size);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.as_ptr().add(offset), data.len());
        }
    }

    /// Read a byte from memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure `offset < self.size()`.
    #[must_use]
    pub unsafe fn read_u8(&self, offset: usize) -> u8 {
        debug_assert!(offset < self.memory_size);
        unsafe { *self.as_ptr().add(offset) }
    }

    /// Write a byte to memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure `offset < self.size()`.
    pub unsafe fn write_u8(&mut self, offset: usize, value: u8) {
        debug_assert!(offset < self.memory_size);
        unsafe { *self.as_ptr().add(offset) = value };
    }
}

impl Drop for GuardedMemory {
    fn drop(&mut self) {
        unsafe {
            let _ = munmap(self.region, self.total_size);
        }
    }
}

// GuardedMemory is Send but not Sync (contains raw pointer)
unsafe impl Send for GuardedMemory {}
