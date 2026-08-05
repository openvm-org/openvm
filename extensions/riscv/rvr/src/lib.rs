//! RV64 extensions for rvr-openvm.
//!
//! Provides opcode lifters, C code generation, and runtime hooks for RV64I,
//! RV64M, RV64 IO, and RV64-specific phantom instructions.
#![cfg(feature = "rvr")]

mod i;
mod instruction;
mod io;
mod m;
mod phantom;

pub use i::RiscvIExtension;
pub use io::{RiscvIoExtension, RiscvIoRuntimeHooks};
pub use m::RiscvMExtension;
pub use phantom::{RiscvPhantomExtension, RiscvPhantomRuntimeHooks};
