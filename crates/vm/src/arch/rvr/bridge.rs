//! Buffer access between [`VmState`] and the rvr FFI layer.
//!
//! Memory and public-values bytes are aliased via raw pointer; registers are
//! the only field still copied (a 128-byte memcpy at execution boundaries).
//! `Streams<F>` and the host RNG are borrowed directly into [`OpenVmIoState`]
//! and never converted upfront.

use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};
use rvr_state::NUM_REGS_I;

use super::ExecuteError;
use crate::{
    arch::{ExecutionError, VmState},
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS,
        online::{GuestMemory, LinearMemory},
        AddressMap,
    },
};

/// Mut pointer to the RV32 main memory address space inside `vm_state`.
/// The pointer is stable for the lifetime of `vm_state.memory`'s backing.
pub fn rv32_memory_ptr<F>(vm_state: &mut VmState<F, GuestMemory>) -> *mut u8 {
    vm_state.memory.memory.mem[RV32_MEMORY_AS as usize]
        .as_mut_slice()
        .as_mut_ptr()
}

pub fn public_values_slice(memory: &mut AddressMap) -> &mut [u8] {
    memory.mem[PUBLIC_VALUES_AS as usize].as_mut_slice()
}

pub fn read_rv32_registers<F>(vm_state: &VmState<F, GuestMemory>) -> [u32; NUM_REGS_I] {
    let bytes = vm_state.memory.memory.mem[RV32_REGISTER_AS as usize].as_slice();
    let mut regs = [0u32; NUM_REGS_I];
    for (reg, chunk) in regs.iter_mut().zip(bytes.chunks_exact(4)) {
        *reg = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    regs
}

pub fn write_rv32_registers<F>(vm_state: &mut VmState<F, GuestMemory>, regs: &[u32; NUM_REGS_I]) {
    let bytes = vm_state.memory.memory.mem[RV32_REGISTER_AS as usize].as_mut_slice();
    for (reg, dst) in regs.iter().zip(bytes.chunks_exact_mut(4)) {
        dst.copy_from_slice(&reg.to_le_bytes());
    }
}

pub fn map_rvr_execute_error(err: ExecuteError) -> ExecutionError {
    match err {
        ExecuteError::GuestExit(code) => ExecutionError::FailedWithExitCode(code as u32),
        other => ExecutionError::RvrExecution(other.to_string()),
    }
}

pub fn map_rvr_compile_error(err: super::compile::CompileError) -> crate::arch::StaticProgramError {
    crate::arch::StaticProgramError::FailToGenerateDynamicLibrary {
        err: err.to_string(),
    }
}
