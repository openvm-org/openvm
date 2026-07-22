//! Machine state shared with the generated rvr-openvm runtime.

mod instret;
mod state;

pub use instret::InstretTrackingState;
pub use state::{ExecutionStatus, RvState, NUM_REGS};
