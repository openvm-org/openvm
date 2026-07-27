//! C-compatible logical execution events derived by GPU replay.

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
    pub value: [u16; 4],
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
/// Register candidates are emitted only for a first-event write. Other address
/// spaces may append one candidate per write for cold finalization to compact.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightInitialWrite {
    pub address_space: u32,
    pub pointer: u32,
    pub initial_value: [u16; 4],
}

/// Four native field cells in the raw representation used by the prover.
///
/// Field-valued accesses store an index into a dense sidecar in the compact
/// `value` payload of [`PreflightMemoryEvent`] and
/// [`PreflightInitialWrite`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreflightFieldBlock {
    pub values: [u32; 4],
}

const _: () = {
    assert!(size_of::<PreflightProgramEvent>() == 8);
    assert!(align_of::<PreflightProgramEvent>() == 4);
    assert!(size_of::<PreflightMemoryEvent>() == 20);
    assert!(align_of::<PreflightMemoryEvent>() == 4);
    assert!(size_of::<PreflightInitialWrite>() == 16);
    assert!(align_of::<PreflightInitialWrite>() == 4);
    assert!(size_of::<PreflightFieldBlock>() == 16);
    assert!(align_of::<PreflightFieldBlock>() == 4);
};
