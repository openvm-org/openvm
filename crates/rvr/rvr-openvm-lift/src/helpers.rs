use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use openvm_instructions::VmOpcode;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::RvrExtensionCtx;

/// Resolve an opcode to its AIR index using the executor inventory.
pub fn resolve_opcode_air_idx(opcode: VmOpcode, ctx: &RvrExtensionCtx) -> u32 {
    ctx.require_opcode_air_idx(opcode)
}

/// Decode register index from an OpenVM operand.
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (f.as_canonical_u32() / RV32_REGISTER_NUM_LIMBS as u32) as u8
}
