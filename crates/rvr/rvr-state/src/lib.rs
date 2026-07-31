//! Machine state shared with the generated rvr-openvm runtime.

mod instret;
mod preflight_history;
mod preflight_transcript;
mod state;

pub use instret::InstretTrackingState;
pub use preflight_history::{
    PreflightFieldBlock, PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent,
    PREFLIGHT_ADDRESS_SPACE_MASK, PREFLIGHT_WRITE_BIT,
};
pub use preflight_transcript::{
    PreflightTranscriptState, RvrCheckpoint, PREFLIGHT_DIRTY_PAGE_BYTES,
};
pub use state::{ExecutionStatus, RvState, NUM_REGS};
