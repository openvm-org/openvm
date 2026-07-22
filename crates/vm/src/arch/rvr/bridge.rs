//! Buffer access between [`VmState`] and the rvr FFI layer.
//!
//! Memory and public-values bytes are aliased via raw pointer; registers are
//! the only field still copied (a 256-byte memcpy at execution boundaries).
//! `Streams` and the host RNG are borrowed directly into [`OpenVmIoState`]
//! and never converted upfront.

use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_NUM_REGISTERS, RV64_REGISTER_AS, RV64_REGISTER_BYTES},
    DEFERRAL_AS, PUBLIC_VALUES_AS,
};
use rvr_state::NUM_REGS;

use super::{compile::CompileError, ExecuteError};
use crate::{
    arch::{ExecutionError, StaticProgramError, VmState},
    system::memory::{
        online::{GuestMemory, LinearMemory},
        AddressMap,
    },
};

const _: () = {
    assert!(NUM_REGS == RV64_NUM_REGISTERS);
    assert!(core::mem::size_of::<u64>() == RV64_REGISTER_BYTES as usize);
};

/// Mut pointer to the RV64 main memory address space inside `vm_state`.
/// The pointer is stable for the lifetime of `vm_state.memory`'s backing.
pub fn rv64_memory_ptr(vm_state: &mut VmState<GuestMemory>) -> *mut u8 {
    vm_state.memory.memory.mem[RV64_MEMORY_AS as usize]
        .as_mut_slice()
        .as_mut_ptr()
}

pub fn public_values_slice(memory: &mut AddressMap) -> &mut [u8] {
    memory.mem[PUBLIC_VALUES_AS as usize].as_mut_slice()
}

pub fn deferral_memory_ptr(memory: &mut AddressMap) -> (*mut u8, usize) {
    let bytes = memory.mem[DEFERRAL_AS as usize].as_mut_slice();
    (bytes.as_mut_ptr(), bytes.len())
}

pub fn read_rv64_registers(vm_state: &VmState<GuestMemory>) -> [u64; NUM_REGS] {
    let bytes = vm_state.memory.memory.mem[RV64_REGISTER_AS as usize].as_slice();
    let mut regs = [0u64; NUM_REGS];
    for (reg, chunk) in regs
        .iter_mut()
        .zip(bytes.chunks_exact(RV64_REGISTER_BYTES as usize))
    {
        *reg = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    regs
}

pub fn write_rv64_registers(vm_state: &mut VmState<GuestMemory>, regs: &[u64; NUM_REGS]) {
    let bytes = vm_state.memory.memory.mem[RV64_REGISTER_AS as usize].as_mut_slice();
    for (reg, dst) in regs
        .iter()
        .zip(bytes.chunks_exact_mut(RV64_REGISTER_BYTES as usize))
    {
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
