//! C-compatible state and transcript buffers used by compiled preflight.
//!
//! The hot-path ABI has two append-only outputs: architectural checkpoints and
//! replay values. Dirty-page words are separate host-to-device transfer
//! metadata and are not part of the replay transcript.

use core::mem::{align_of, offset_of, size_of};

/// Byte size of one sparse-upload dirty page.
///
/// The VM asserts this against its host-to-device `TouchedPages` page size.
pub const PREFLIGHT_DIRTY_PAGE_BYTES: usize = 4096;

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
    pub replay_value_cursor: u32,
    pub regs: [u64; 31],
}

/// Raw buffer descriptors shared with generated preflight C.
///
/// The pointers refer to spare capacity owned by Rust for the duration of one
/// execution call. Generated code updates only lengths and scalar execution
/// state. Checkpoint and replay-value cursors deliberately remain here instead of
/// consuming registers in the generated block ABI.
#[repr(C)]
#[derive(Debug)]
pub struct PreflightTranscriptState {
    pub checkpoint_log: *mut RvrCheckpoint,
    pub replay_value_log: *mut u64,
    pub checkpoint_log_len: u64,
    pub checkpoint_log_cap: u64,
    pub replay_value_log_len: u64,
    pub replay_value_log_cap: u64,
    pub timestamp: u32,
    pub retired: u32,
    pub checkpoint_interval: u32,
    pub last_checkpoint_retired: u32,
    pub error: u32,
    pub instruction_limit: u32,
    pub memory_dirty_pages: *mut u64,
    pub memory_dirty_page_words: u64,
    pub last_memory_dirty_page: u32,
    pub padding: u32,
}

impl Default for PreflightTranscriptState {
    fn default() -> Self {
        Self {
            checkpoint_log: core::ptr::null_mut(),
            replay_value_log: core::ptr::null_mut(),
            checkpoint_log_len: 0,
            checkpoint_log_cap: 0,
            replay_value_log_len: 0,
            replay_value_log_cap: 0,
            timestamp: 1,
            retired: 0,
            checkpoint_interval: 0,
            last_checkpoint_retired: 0,
            error: 0,
            instruction_limit: u32::MAX,
            memory_dirty_pages: core::ptr::null_mut(),
            memory_dirty_page_words: 0,
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
    assert!(offset_of!(RvrCheckpoint, replay_value_cursor) == 12);
    assert!(offset_of!(RvrCheckpoint, regs) == 16);

    assert!(size_of::<PreflightTranscriptState>() == 96);
    assert!(align_of::<PreflightTranscriptState>() == 8);
    assert!(offset_of!(PreflightTranscriptState, checkpoint_log) == 0);
    assert!(offset_of!(PreflightTranscriptState, replay_value_log) == 8);
    assert!(offset_of!(PreflightTranscriptState, checkpoint_log_len) == 16);
    assert!(offset_of!(PreflightTranscriptState, checkpoint_log_cap) == 24);
    assert!(offset_of!(PreflightTranscriptState, replay_value_log_len) == 32);
    assert!(offset_of!(PreflightTranscriptState, replay_value_log_cap) == 40);
    assert!(offset_of!(PreflightTranscriptState, timestamp) == 48);
    assert!(offset_of!(PreflightTranscriptState, retired) == 52);
    assert!(offset_of!(PreflightTranscriptState, checkpoint_interval) == 56);
    assert!(offset_of!(PreflightTranscriptState, last_checkpoint_retired) == 60);
    assert!(offset_of!(PreflightTranscriptState, error) == 64);
    assert!(offset_of!(PreflightTranscriptState, instruction_limit) == 68);
    assert!(offset_of!(PreflightTranscriptState, memory_dirty_pages) == 72);
    assert!(offset_of!(PreflightTranscriptState, memory_dirty_page_words) == 80);
    assert!(offset_of!(PreflightTranscriptState, last_memory_dirty_page) == 88);
    assert!(offset_of!(PreflightTranscriptState, padding) == 92);
};
