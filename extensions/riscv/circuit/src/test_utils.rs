use openvm_circuit::arch::{
    testing::{memory::gen_register_pointer, TestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};

use super::adapters::{rv64_bytes_to_u16_block, RV64_REGISTER_NUM_LIMBS, RV_IS_TYPE_IMM_BITS};

// Returns (instruction, rd)
#[cfg_attr(all(feature = "test-utils", not(test)), allow(dead_code))]
pub fn rv64_rand_write_register_or_imm(
    tester: &mut impl TestBuilder<BabyBear>,
    rs1_writes: [u8; RV64_REGISTER_NUM_LIMBS],
    rs2_writes: [u8; RV64_REGISTER_NUM_LIMBS],
    imm: Option<usize>,
    opcode_with_offset: usize,
    rng: &mut StdRng,
) -> (Instruction<BabyBear>, usize) {
    let rs2_is_imm = imm.is_some();

    let rs1 = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2 = imm.unwrap_or_else(|| gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS));
    let rd = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs1, rs1_writes.map(BabyBear::from_u8));
    if !rs2_is_imm {
        tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs2, rs2_writes.map(BabyBear::from_u8));
    }

    (
        Instruction::from_usize(
            VmOpcode::from_usize(opcode_with_offset),
            [rd, rs1, rs2, 1, if rs2_is_imm { 0 } else { 1 }],
        ),
        rd,
    )
}

#[cfg_attr(all(feature = "test-utils", not(test)), allow(dead_code))]
pub fn generate_rv64_is_type_immediate(rng: &mut StdRng) -> (usize, [u8; RV64_REGISTER_NUM_LIMBS]) {
    let mut imm: u32 = rng.random_range(0..(1 << RV_IS_TYPE_IMM_BITS));
    if (imm & 0x800) != 0 {
        imm |= !0xFFF
    }
    let sign_byte = (imm >> 16) as u8;
    (
        (imm & 0xFFFFFF) as usize,
        [
            imm as u8,
            (imm >> 8) as u8,
            sign_byte,
            sign_byte,
            sign_byte,
            sign_byte,
            sign_byte,
            sign_byte,
        ],
    )
}

#[cfg_attr(all(feature = "test-utils", not(test)), allow(dead_code))]
pub fn rv64_marker_bytes_to_u16_marker(
    marker: [u8; RV64_REGISTER_NUM_LIMBS],
) -> [u32; BLOCK_FE_WIDTH] {
    rv64_bytes_to_u16_block(marker).map(|marker| u32::from(marker != 0))
}

#[cfg_attr(all(feature = "test-utils", not(test)), allow(dead_code))]
pub fn rv64_msb_byte_prank_to_u16_limb(bytes: [u8; RV64_REGISTER_NUM_LIMBS], msb: i32) -> i32 {
    let low_byte = bytes[RV64_REGISTER_NUM_LIMBS - 2];
    let limb = u16::from_le_bytes([low_byte, msb as u8]);
    if msb < 0 {
        -i32::from(limb.wrapping_neg())
    } else {
        i32::from(limb)
    }
}
