use ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::{
    arch::testing::{memory::gen_pointer, VmChipTestBuilder},
    ax_stark_sdk::p3_baby_bear::BabyBear,
    rv32im::adapters::{RV32_REGISTER_NUM_LIMBS, RV_IS_TYPE_IMM_BITS},
};
use axvm_instructions::instruction::Instruction;
use rand::{rngs::StdRng, Rng};

// Returns (instruction, rd)
pub fn rv32_rand_write_register_or_imm<const NUM_LIMBS: usize>(
    tester: &mut VmChipTestBuilder<BabyBear>,
    rs1_writes: [u32; NUM_LIMBS],
    rs2_writes: [u32; NUM_LIMBS],
    imm: Option<usize>,
    opcode_with_offset: usize,
    rng: &mut StdRng,
) -> (Instruction<BabyBear>, usize) {
    let rs2_is_imm = imm.is_some();

    let rs1 = gen_pointer(rng, NUM_LIMBS);
    let rs2 = imm.unwrap_or_else(|| gen_pointer(rng, NUM_LIMBS));
    let rd = gen_pointer(rng, NUM_LIMBS);

    tester.write::<NUM_LIMBS>(1, rs1, rs1_writes.map(BabyBear::from_canonical_u32));
    if !rs2_is_imm {
        tester.write::<NUM_LIMBS>(1, rs2, rs2_writes.map(BabyBear::from_canonical_u32));
    }

    (
        Instruction::from_usize(
            opcode_with_offset,
            [rd, rs1, rs2, 1, if rs2_is_imm { 0 } else { 1 }],
        ),
        rd,
    )
}

pub fn generate_rv32_is_type_immediate(
    rng: &mut StdRng,
) -> (usize, [u32; RV32_REGISTER_NUM_LIMBS]) {
    let mut imm: u32 = rng.gen_range(0..(1 << RV_IS_TYPE_IMM_BITS));
    if (imm & 0x800) != 0 {
        imm |= !0xFFF
    }
    (
        (imm & 0xFFFFFF) as usize,
        [
            imm as u8,
            (imm >> 8) as u8,
            (imm >> 16) as u8,
            (imm >> 16) as u8,
        ]
        .map(|x| x as u32),
    )
}
