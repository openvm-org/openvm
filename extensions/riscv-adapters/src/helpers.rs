use openvm_circuit::system::memory::online::TracingMemory;
use openvm_instructions::riscv::RV64_REGISTER_AS;
use openvm_riscv_circuit::adapters::{rv64_bytes_to_u32, tracing_read, RV64_REGISTER_NUM_LIMBS};

/// Reads an 8-byte register, debug-asserts that the upper 4 bytes are zero (enforced at proving
/// time by zero padding in the memory bus interaction), and returns the low 4 bytes as a u32
/// pointer.
#[inline(always)]
pub(crate) fn tracing_read_reg_ptr(
    memory: &mut TracingMemory,
    ptr: u32,
    prev_timestamp: &mut u32,
) -> u32 {
    let bytes: [u8; RV64_REGISTER_NUM_LIMBS] =
        tracing_read(memory, RV64_REGISTER_AS, ptr, prev_timestamp);
    rv64_bytes_to_u32(bytes)
}
