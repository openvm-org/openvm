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

/// Mut pointer + size of the RV32 main memory address space inside `vm_state`.
///
/// The pointer is stable for the lifetime of `vm_state.memory`'s backing.
pub fn rv32_memory_ptr<F>(vm_state: &mut VmState<F, GuestMemory>) -> (*mut u8, usize) {
    let mem = &mut vm_state.memory.memory.mem[RV32_MEMORY_AS as usize];
    let len = mem.size();
    (mem.as_mut_slice().as_mut_ptr(), len)
}

/// Mutable byte slice for the public-values address space.
pub fn public_values_slice(memory: &mut AddressMap) -> &mut [u8] {
    memory.mem[PUBLIC_VALUES_AS as usize].as_mut_slice()
}

/// Read the 32 RV32 GPRs from `vm_state`'s register address space.
pub fn read_rv32_registers<F>(vm_state: &VmState<F, GuestMemory>) -> [u32; NUM_REGS_I] {
    let mem = &vm_state.memory.memory.mem[RV32_REGISTER_AS as usize];
    let bytes = mem.as_slice();
    let mut regs = [0u32; NUM_REGS_I];
    for (i, chunk) in bytes.chunks_exact(4).take(NUM_REGS_I).enumerate() {
        regs[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    regs
}

/// Write the 32 RV32 GPRs back into `vm_state`'s register address space.
pub fn write_rv32_registers<F>(vm_state: &mut VmState<F, GuestMemory>, regs: &[u32; NUM_REGS_I]) {
    let mem = &mut vm_state.memory.memory.mem[RV32_REGISTER_AS as usize];
    let bytes = mem.as_mut_slice();
    for (i, reg) in regs.iter().enumerate() {
        let dst = &mut bytes[i * 4..i * 4 + 4];
        dst.copy_from_slice(&reg.to_le_bytes());
    }
}

/// Map an rvr execute error into the openvm-circuit `ExecutionError` enum.
pub fn map_rvr_execute_error(err: ExecuteError) -> ExecutionError {
    match err {
        ExecuteError::GuestExit(code) => ExecutionError::FailedWithExitCode(code as u32),
        other => ExecutionError::RvrExecution(other.to_string()),
    }
}

/// Map an rvr compile error into the openvm-circuit `StaticProgramError` enum.
pub fn map_rvr_compile_error(err: super::compile::CompileError) -> crate::arch::StaticProgramError {
    crate::arch::StaticProgramError::FailToGenerateDynamicLibrary {
        err: err.to_string(),
    }
}

/// Validate the rvr execution outcome and translate to `ExecutionError`.
pub fn ensure_rvr_outcome(
    context: &str,
    terminated: bool,
    suspended: bool,
    guest_exit_code: u8,
    allow_suspended: bool,
) -> Result<(), ExecutionError> {
    if allow_suspended && suspended {
        return Ok(());
    }
    if terminated {
        if guest_exit_code == 0 {
            return Ok(());
        }
        return Err(ExecutionError::FailedWithExitCode(guest_exit_code as u32));
    }
    Err(ExecutionError::RvrExecution(format!(
        "{context} failed: terminated={terminated}, suspended={suspended}, guest_exit_code={guest_exit_code}"
    )))
}
