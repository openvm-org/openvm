use openvm_circuit::arch::testing::{memory::gen_pointer, TestBuilder};
use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_rv32im_circuit::adapters::{RV32_REGISTER_NUM_LIMBS, RV_IS_TYPE_IMM_BITS};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};

pub fn write_ptr_reg(
    tester: &mut impl TestBuilder<BabyBear>,
    ptr_as: usize,
    reg_addr: usize,
    value: u32,
) {
    tester.write(
        ptr_as,
        reg_addr,
        value.to_le_bytes().map(BabyBear::from_canonical_u8),
    );
}

pub fn rv32_write_heap_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) =
        tester.write_heap_default::<NUM_LIMBS>(RV32_REGISTER_NUM_LIMBS, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) =
            tester.write_heap_default::<NUM_LIMBS>(RV32_REGISTER_NUM_LIMBS, 128, addr2_writes);
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(RV32_REGISTER_NUM_LIMBS, 128);

    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        reg3 as isize,
        reg1 as isize,
        reg2 as isize,
        1_isize,
        2_isize,
    )
}

pub fn rv32_write_heap_default_with_increment<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    pointer_increment: usize,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) = tester.write_heap_default::<NUM_LIMBS>(
        RV32_REGISTER_NUM_LIMBS,
        pointer_increment,
        addr1_writes,
    );
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) = tester.write_heap_default::<NUM_LIMBS>(
            RV32_REGISTER_NUM_LIMBS,
            pointer_increment,
            addr2_writes,
        );
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(RV32_REGISTER_NUM_LIMBS, pointer_increment);

    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        reg3 as isize,
        reg1 as isize,
        reg2 as isize,
        1_isize,
        2_isize,
    )
}

pub fn rv32_heap_branch_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    imm: isize,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) =
        tester.write_heap_default::<NUM_LIMBS>(RV32_REGISTER_NUM_LIMBS, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) =
            tester.write_heap_default::<NUM_LIMBS>(RV32_REGISTER_NUM_LIMBS, 128, addr2_writes);
        reg2
    };

    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        reg1 as isize,
        reg2 as isize,
        imm,
        1_isize,
        2_isize,
    )
}

// Returns (instruction, rd)
pub fn rv32_rand_write_register_or_imm<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
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
            VmOpcode::from_usize(opcode_with_offset),
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
