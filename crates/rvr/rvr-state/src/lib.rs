//! RV64 machine state for the rvr-openvm runtime.
//!
//! Provides `RvState` (a Rust struct whose layout matches the generated C
//! `RvState`), guarded memory, and the marker traits that let `rvr-openvm`
//! plug in its own tracer/suspender types.

mod memory;
mod state;
mod suspender;
mod tracer;
mod xlen;

pub use memory::{GuardedMemory, MemoryError, GUARD_SIZE};
pub use state::{ExecutionStatus, Rv64State, RvState, NUM_CSRS, NUM_REGS_I};
pub use suspender::{InstretSuspender, SuspenderState};
pub use tracer::TracerState;
pub use xlen::{Rv64, Xlen};
