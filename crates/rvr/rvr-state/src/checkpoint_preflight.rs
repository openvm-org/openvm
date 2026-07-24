//! C-compatible buffers used by experimental checkpoint preflight execution.

use core::mem::{align_of, offset_of, size_of};

/// Byte size of one sparse-upload dirty page.
///
/// The VM asserts this against its host-to-device `TouchedPages` page size.
pub const CHECKPOINT_DIRTY_PAGE_BYTES: usize = 4096;

/// One independently replayable execution boundary.
///
/// Register x0 is implicit. The stored register array is x1 through x31 in
/// architectural order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RvrCheckpoint {
    pub pc: u32,
    pub timestamp: u32,
    pub retired: u32,
    pub residual_cursor: u32,
    pub regs: [u64; 31],
}

/// Raw buffer descriptors shared with generated checkpoint-preflight C.
///
/// The pointers refer to spare capacity owned by Rust for the duration of one
/// execution call. Generated code updates only lengths and scalar execution
/// state. Checkpoint and residual cursors deliberately remain here instead of
/// consuming registers in the generated block ABI.
#[repr(C)]
#[derive(Debug)]
pub struct CheckpointPreflightState {
    pub checkpoint_log: *mut RvrCheckpoint,
    pub residual_log: *mut u64,
    pub checkpoint_log_len: u64,
    pub checkpoint_log_cap: u64,
    pub residual_log_len: u64,
    pub residual_log_cap: u64,
    pub timestamp: u32,
    pub retired: u32,
    pub checkpoint_interval: u32,
    pub last_checkpoint_retired: u32,
    pub error: u32,
    pub instruction_limit: u32,
    pub memory_dirty_pages: *mut u64,
    pub public_values_dirty_pages: *mut u64,
    pub memory_dirty_page_words: u64,
    pub public_values_dirty_page_words: u64,
    pub last_memory_dirty_page: u32,
    pub padding: u32,
}

impl Default for CheckpointPreflightState {
    fn default() -> Self {
        Self {
            checkpoint_log: core::ptr::null_mut(),
            residual_log: core::ptr::null_mut(),
            checkpoint_log_len: 0,
            checkpoint_log_cap: 0,
            residual_log_len: 0,
            residual_log_cap: 0,
            timestamp: 1,
            retired: 0,
            checkpoint_interval: 0,
            last_checkpoint_retired: 0,
            error: 0,
            instruction_limit: u32::MAX,
            memory_dirty_pages: core::ptr::null_mut(),
            public_values_dirty_pages: core::ptr::null_mut(),
            memory_dirty_page_words: 0,
            public_values_dirty_page_words: 0,
            last_memory_dirty_page: u32::MAX,
            padding: 0,
        }
    }
}

const _: () = {
    assert!(size_of::<RvrCheckpoint>() == 264);
    assert!(align_of::<RvrCheckpoint>() == 8);
    assert!(offset_of!(RvrCheckpoint, pc) == 0);
    assert!(offset_of!(RvrCheckpoint, timestamp) == 4);
    assert!(offset_of!(RvrCheckpoint, retired) == 8);
    assert!(offset_of!(RvrCheckpoint, residual_cursor) == 12);
    assert!(offset_of!(RvrCheckpoint, regs) == 16);

    assert!(size_of::<CheckpointPreflightState>() == 112);
    assert!(align_of::<CheckpointPreflightState>() == 8);
    assert!(offset_of!(CheckpointPreflightState, checkpoint_log) == 0);
    assert!(offset_of!(CheckpointPreflightState, residual_log) == 8);
    assert!(offset_of!(CheckpointPreflightState, checkpoint_log_len) == 16);
    assert!(offset_of!(CheckpointPreflightState, checkpoint_log_cap) == 24);
    assert!(offset_of!(CheckpointPreflightState, residual_log_len) == 32);
    assert!(offset_of!(CheckpointPreflightState, residual_log_cap) == 40);
    assert!(offset_of!(CheckpointPreflightState, timestamp) == 48);
    assert!(offset_of!(CheckpointPreflightState, retired) == 52);
    assert!(offset_of!(CheckpointPreflightState, checkpoint_interval) == 56);
    assert!(offset_of!(CheckpointPreflightState, last_checkpoint_retired) == 60);
    assert!(offset_of!(CheckpointPreflightState, error) == 64);
    assert!(offset_of!(CheckpointPreflightState, instruction_limit) == 68);
    assert!(offset_of!(CheckpointPreflightState, memory_dirty_pages) == 72);
    assert!(offset_of!(CheckpointPreflightState, public_values_dirty_pages) == 80);
    assert!(offset_of!(CheckpointPreflightState, memory_dirty_page_words) == 88);
    assert!(offset_of!(CheckpointPreflightState, public_values_dirty_page_words) == 96);
    assert!(offset_of!(CheckpointPreflightState, last_memory_dirty_page) == 104);
    assert!(offset_of!(CheckpointPreflightState, padding) == 108);
};
