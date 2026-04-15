//! RV32 machine state for the rvr-openvm runtime.
//!
//! Provides `RvState` (a Rust struct whose layout matches the generated C
//! `RvState`), guarded memory, and the marker traits that let `rvr-openvm`
//! plug in its own tracer/suspender types.

mod memory;
mod state;
mod suspender;
mod tracer;
mod xlen;

pub use memory::{GUARD_SIZE, GuardedMemory, MemoryError};
pub use state::{ExecutionStatus, NUM_CSRS, NUM_REGS_I, Rv32State, RvState};
pub use suspender::{InstretSuspender, SuspenderState};
pub use tracer::TracerState;
pub use xlen::{Rv32, Xlen};
