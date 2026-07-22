//! Machine state shared with the generated rvr-openvm runtime.

mod instret;
mod preflight;
mod state;

pub use instret::InstretTrackingState;
pub use preflight::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PreflightState,
    PREFLIGHT_ADDRESS_SPACE_MASK, PREFLIGHT_WRITE_BIT,
};
pub use state::{ExecutionStatus, RvState, NUM_REGS};
