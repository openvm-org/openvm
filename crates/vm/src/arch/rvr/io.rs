//! OpenVM IO runtime: ctx (`OpenVmIoState`) borrowed from `VmState`.

use std::{collections::VecDeque, ffi::c_void, ops::Range};

#[cfg(not(feature = "unprotected"))]
use openvm_platform::memory::MEM_SIZE;
use rand::rngs::StdRng;
use rvr_state::CHECKPOINT_DIRTY_PAGE_BYTES;

use crate::arch::{deferral::DeferralState, HintStream};

/// IO execution state borrowed from the host `VmState` for the duration of
/// one rvr call. Streams, rng, and the public-values byte slice are mutable
/// borrows; `memory_ptr` is a raw alias of VmState's main memory buffer
/// (raw because the C engine accesses it directly via pointer).
pub struct OpenVmIoState<'a> {
    pub input_stream: &'a mut VecDeque<Vec<u8>>,
    pub hint_stream: &'a mut HintStream,
    pub rng: &'a mut StdRng,
    pub memory_ptr: *mut u8,
    pub public_values: &'a mut [u8],
    pub deferral_memory: *mut u8,
    pub deferral_memory_len_bytes: usize,
    /// Executor-only AS4 dirty-page bits used by checkpoint execution. Other
    /// execution modes leave this absent, so their generated hot paths do no
    /// dirty-page work.
    pub checkpoint_deferral_dirty_pages: Option<&'a mut [u64]>,
    pub deferrals: &'a mut Vec<DeferralState>,
}

impl OpenVmIoState<'_> {
    /// Returns whether checkpoint dirty-page storage can represent an AS4
    /// write. Non-checkpoint modes have no bitmap and are always valid.
    #[inline]
    pub fn can_mark_checkpoint_deferral_write(&self, byte_start: usize, byte_len: usize) -> bool {
        let Some(dirty_pages) = self.checkpoint_deferral_dirty_pages.as_deref() else {
            return true;
        };
        if byte_len == 0 {
            return true;
        }
        let Some(byte_end) = byte_start.checked_add(byte_len - 1) else {
            return false;
        };
        let last_page = byte_end / CHECKPOINT_DIRTY_PAGE_BYTES;
        last_page / (u64::BITS as usize) < dirty_pages.len()
    }

    /// Marks an AS4 byte range written by a host callback. Deferral CALL is the
    /// only proof-visible AS4 writer that bypasses generated C. The caller must
    /// validate the range before mutating AS4 so this commit step cannot fail.
    #[inline]
    pub fn mark_checkpoint_deferral_write(&mut self, byte_start: usize, byte_len: usize) {
        debug_assert!(self.can_mark_checkpoint_deferral_write(byte_start, byte_len));
        let Some(dirty_pages) = self.checkpoint_deferral_dirty_pages.as_deref_mut() else {
            return;
        };
        if byte_len == 0 {
            return;
        }

        let Some(byte_end) = byte_start.checked_add(byte_len - 1) else {
            return;
        };
        let first_page = byte_start / CHECKPOINT_DIRTY_PAGE_BYTES;
        let last_page = byte_end / CHECKPOINT_DIRTY_PAGE_BYTES;
        for page in first_page..=last_page {
            let word = page / u64::BITS as usize;
            let Some(word_bits) = dirty_pages.get_mut(word) else {
                debug_assert!(false, "prevalidated checkpoint dirty range became invalid");
                return;
            };
            *word_bits |= 1 << (page % u64::BITS as usize);
        }
    }
}

/// Return host indices for an RV64 range that fits in `AS_MEMORY` (`MEM_SIZE`
/// bytes).
#[cfg(not(feature = "unprotected"))]
#[inline(always)]
pub fn checked_mem_bounds_range(start: u64, num_bytes: u64) -> Option<Range<usize>> {
    let end = start.checked_add(num_bytes)?;
    (end <= MEM_SIZE as u64).then_some(start as usize..end as usize)
}

#[cfg(feature = "unprotected")]
#[inline(always)]
pub fn checked_mem_bounds_range(start: u64, num_bytes: u64) -> Option<Range<usize>> {
    let start = start as usize;
    Some(start..start.wrapping_add(num_bytes as usize))
}

/// Replace the hint stream contents. Called via `ext_hint_stream_set` from extension FFI.
///
/// # Safety
///
/// `ctx` must be a valid `OpenVmIoState` pointer. `data` must point to `len` bytes (or be null).
pub unsafe extern "C" fn host_hint_stream_set(ctx: *mut c_void, data: *const u8, len: u64) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if len > 0 && !data.is_null() {
        let len = usize::try_from(len).expect("hint stream length must fit in usize");
        let slice = unsafe { std::slice::from_raw_parts(data, len) };
        io.hint_stream.set_hint_from_slice(slice);
    } else {
        io.hint_stream.clear();
    }
}

#[cfg(all(test, not(feature = "unprotected")))]
mod tests {
    use super::*;

    #[test]
    fn checked_memory_range_accepts_end_boundary() {
        assert_eq!(
            checked_mem_bounds_range(MEM_SIZE as u64 - 8, 8),
            Some(MEM_SIZE - 8..MEM_SIZE)
        );
        assert_eq!(
            checked_mem_bounds_range(MEM_SIZE as u64, 0),
            Some(MEM_SIZE..MEM_SIZE)
        );
    }

    #[test]
    fn checked_memory_range_rejects_out_of_bounds_and_overflow() {
        assert_eq!(checked_mem_bounds_range(MEM_SIZE as u64, 1), None);
        assert_eq!(checked_mem_bounds_range(u64::MAX, 1), None);
    }
}
