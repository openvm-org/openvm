//! Buffer access between [`VmState`] and the rvr FFI layer.
//!
//! Memory and public-values bytes are aliased via raw pointer; registers are
//! the only field still copied (a 256-byte memcpy at execution boundaries).
//! `Streams` and the host RNG are borrowed directly into [`OpenVmIoState`]
//! and never converted upfront.

use std::mem::{align_of, size_of};

use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS,
};
use rvr_state::NUM_REGS_I;

use super::{compile::CompileError, ExecuteError};
use crate::{
    arch::{ExecutionError, StaticProgramError, VmState},
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS,
        online::{GuestMemory, LinearMemory},
        AddressMap,
    },
};

/// Mut pointer to the RV64 main memory address space inside `vm_state`.
/// The pointer is stable for the lifetime of `vm_state.memory`'s backing.
pub fn rv64_memory_ptr<F>(vm_state: &mut VmState<F, GuestMemory>) -> *mut u8 {
    vm_state.memory.memory.mem[RV64_MEMORY_AS as usize]
        .as_mut_slice()
        .as_mut_ptr()
}

pub fn public_values_slice(memory: &mut AddressMap) -> &mut [u8] {
    memory.mem[PUBLIC_VALUES_AS as usize].as_mut_slice()
}

/// Raw alias of the `F`-typed DEFERRAL address space.
/// The returned length is in `F` cells, not bytes.
pub fn deferral_memory_ptr<F>(memory: &mut AddressMap) -> (*mut F, usize) {
    let bytes = memory.mem[DEFERRAL_AS as usize].as_mut_slice();
    debug_assert_eq!(
        bytes.as_mut_ptr().addr() % align_of::<F>(),
        0,
        "DEFERRAL_AS buffer must be aligned for F"
    );
    debug_assert_eq!(bytes.len() % size_of::<F>(), 0);
    (bytes.as_mut_ptr().cast::<F>(), bytes.len() / size_of::<F>())
}

pub fn read_rv64_registers<F>(vm_state: &VmState<F, GuestMemory>) -> [u64; NUM_REGS_I] {
    let bytes = vm_state.memory.memory.mem[RV64_REGISTER_AS as usize].as_slice();
    let mut regs = [0u64; NUM_REGS_I];
    for (reg, chunk) in regs.iter_mut().zip(bytes.chunks_exact(8)) {
        *reg = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    regs
}

pub fn write_rv64_registers<F>(vm_state: &mut VmState<F, GuestMemory>, regs: &[u64; NUM_REGS_I]) {
    let bytes = vm_state.memory.memory.mem[RV64_REGISTER_AS as usize].as_mut_slice();
    for (reg, dst) in regs.iter().zip(bytes.chunks_exact_mut(8)) {
        dst.copy_from_slice(&reg.to_le_bytes());
    }
}

pub fn map_rvr_execute_error(err: ExecuteError) -> ExecutionError {
    match err {
        ExecuteError::GuestExit(code) => ExecutionError::FailedWithExitCode(code as u32),
        other => ExecutionError::RvrExecution(other.to_string()),
    }
}

pub fn map_rvr_compile_error(err: CompileError) -> StaticProgramError {
    StaticProgramError::FailToGenerateDynamicLibrary {
        err: err.to_string(),
    }
}
