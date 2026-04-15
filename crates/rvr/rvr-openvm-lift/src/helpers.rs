use openvm_circuit::arch::ExecutorInventory;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_instructions::VmOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

/// Resolve an opcode to its AIR index using the executor inventory.
pub fn resolve_opcode_air_idx<E>(
    opcode: VmOpcode,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
) -> u32 {
    let executor_idx = *inventory
        .instruction_lookup
        .get(&opcode)
        .unwrap_or_else(|| panic!("opcode {opcode:?} not found in executor inventory"));
    executor_idx_to_air_idx[executor_idx as usize] as u32
}

/// Decode register index from an OpenVM operand.
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (f.as_canonical_u32() / RV32_REGISTER_NUM_LIMBS as u32) as u8
}
