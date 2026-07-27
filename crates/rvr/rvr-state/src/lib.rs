//! Machine state shared with the generated rvr-openvm runtime.

mod checkpoint_preflight;
mod instret;
mod preflight;
mod state;

pub use checkpoint_preflight::{
    CheckpointPreflightState, RvrCheckpoint, CHECKPOINT_DIRTY_PAGE_BYTES,
};
pub use instret::InstretTrackingState;
pub use preflight::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
    PREFLIGHT_ADDRESS_SPACE_MASK, PREFLIGHT_WRITE_BIT,
};
pub use state::{ExecutionStatus, RvState, NUM_REGS};
