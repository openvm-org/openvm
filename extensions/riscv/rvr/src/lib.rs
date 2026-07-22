//! RV64 extensions for rvr-openvm.
//!
//! Provides opcode lifters, generated-C support, and runtime hooks for RV64I,
//! RV64M, RV64IO, and RV64-specific phantom instructions.
#![cfg(feature = "rvr")]

mod i;
mod instruction;
mod io;
mod m;
mod phantom;

pub use i::Rv64IExtension;
pub use io::{Rv64IoExtension, Rv64IoRuntimeHooks};
pub use m::Rv64MExtension;
pub use phantom::{Rv64PhantomExtension, Rv64PhantomRuntimeHooks};
