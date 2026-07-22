//! C-compatible append buffers used by RVR preflight execution.

use core::mem::offset_of;

pub const PREFLIGHT_WRITE_BIT: u32 = 1 << 31;
pub const PREFLIGHT_ADDRESS_SPACE_MASK: u32 = !PREFLIGHT_WRITE_BIT;

/// One fetched instruction, or the final execution sentinel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightProgramEvent {
    pub pc: u32,
    pub timestamp: u32,
}

/// One logical OpenVM memory-bus access.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightMemoryEvent {
    pub timestamp: u32,
    pub address_space_and_kind: u32,
    pub pointer: u32,
    pub value: [u32; 4],
}

impl PreflightMemoryEvent {
    #[inline]
    pub const fn address_space(&self) -> u32 {
        self.address_space_and_kind & PREFLIGHT_ADDRESS_SPACE_MASK
    }

    #[inline]
    pub const fn is_write(&self) -> bool {
        self.address_space_and_kind & PREFLIGHT_WRITE_BIT != 0
    }
}

/// Previous value captured before a write.
///
/// The generated executor may append one candidate per write. Rust finalization
/// retains it only when that write is the block's first timed event.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightInitialWrite {
    pub address_space: u32,
    pub pointer: u32,
    pub initial_value: [u32; 4],
}

/// Raw buffer descriptors shared with generated C.
///
/// The pointers refer to spare capacity owned by Rust for the duration of one
/// execution call. Generated code only updates lengths, timestamp, and error.
#[repr(C)]
#[derive(Debug)]
pub struct PreflightState {
    pub program_log: *mut PreflightProgramEvent,
    pub memory_log: *mut PreflightMemoryEvent,
    pub initial_write_log: *mut PreflightInitialWrite,
    pub program_log_len: u64,
    pub program_log_cap: u64,
    pub memory_log_len: u64,
    pub memory_log_cap: u64,
    pub initial_write_log_len: u64,
    pub initial_write_log_cap: u64,
    pub timestamp: u32,
    pub error: u32,
}

impl Default for PreflightState {
    fn default() -> Self {
        Self {
            program_log: core::ptr::null_mut(),
            memory_log: core::ptr::null_mut(),
            initial_write_log: core::ptr::null_mut(),
            program_log_len: 0,
            program_log_cap: 0,
            memory_log_len: 0,
            memory_log_cap: 0,
            initial_write_log_len: 0,
            initial_write_log_cap: 0,
            timestamp: 1,
            error: 0,
        }
    }
}

const _: () = {
    assert!(size_of::<PreflightProgramEvent>() == 8);
    assert!(align_of::<PreflightProgramEvent>() == 4);
    assert!(size_of::<PreflightMemoryEvent>() == 28);
    assert!(align_of::<PreflightMemoryEvent>() == 4);
    assert!(size_of::<PreflightInitialWrite>() == 24);
    assert!(align_of::<PreflightInitialWrite>() == 4);
    assert!(size_of::<PreflightState>() == 80);
    assert!(align_of::<PreflightState>() == 8);
    assert!(offset_of!(PreflightState, program_log) == 0);
    assert!(offset_of!(PreflightState, memory_log) == 8);
    assert!(offset_of!(PreflightState, initial_write_log) == 16);
    assert!(offset_of!(PreflightState, program_log_len) == 24);
    assert!(offset_of!(PreflightState, timestamp) == 72);
    assert!(offset_of!(PreflightState, error) == 76);
};
