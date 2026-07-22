//! Machine state shared with the generated rvr-openvm runtime.

mod instret;
mod preflight;
mod state;

pub use instret::InstretTrackingState;
pub use preflight::{
    PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PreflightState,
};
pub use state::{ExecutionStatus, RvState, NUM_REGS};
