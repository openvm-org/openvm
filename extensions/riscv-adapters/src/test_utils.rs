use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder},
    BLOCK_FE_WIDTH, U16_CELL_SIZE,
};
use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_riscv_circuit::adapters::{RV64_REGISTER_NUM_LIMBS, RV_IS_TYPE_IMM_BITS};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{rngs::StdRng, Rng};

pub fn write_ptr_reg(
    tester: &mut impl TestBuilder<BabyBear>,
    ptr_as: usize,
    reg_addr: usize,
    value: u64,
) {
    tester.write_bytes(ptr_as, reg_addr, value.to_le_bytes().map(BabyBear::from_u8));
}

/// Writes byte-shaped heap operands and returns the corresponding RV64 instruction.
///
/// Use this for adapters whose heap payload is `MEMORY_BLOCK_BYTES` raw bytes per
/// memory-bus message.
pub fn rv64_write_heap_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) =
        tester.write_heap_default::<NUM_LIMBS>(RV64_REGISTER_NUM_LIMBS, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) =
            tester.write_heap_default::<NUM_LIMBS>(RV64_REGISTER_NUM_LIMBS, 128, addr2_writes);
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(RV64_REGISTER_NUM_LIMBS, 128);

    rv64_heap_instruction(reg3, reg1, reg2, opcode_with_offset)
}

/// Writes u16-cell heap operands and returns the corresponding RV64 instruction.
///
/// Use this for adapters whose heap payload is `BLOCK_FE_WIDTH` u16 cells per
/// memory-bus message.
pub fn rv64_write_u16_heap_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) =
        write_u16_heap_default::<NUM_LIMBS>(tester, RV64_REGISTER_NUM_LIMBS, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) =
            write_u16_heap_default::<NUM_LIMBS>(tester, RV64_REGISTER_NUM_LIMBS, 128, addr2_writes);
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(RV64_REGISTER_NUM_LIMBS, 128);

    rv64_heap_instruction(reg3, reg1, reg2, opcode_with_offset)
}

pub fn rv64_write_heap_default_with_increment<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    pointer_increment: usize,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) = tester.write_heap_default::<NUM_LIMBS>(
        RV64_REGISTER_NUM_LIMBS,
        pointer_increment,
        addr1_writes,
    );
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) = tester.write_heap_default::<NUM_LIMBS>(
            RV64_REGISTER_NUM_LIMBS,
            pointer_increment,
            addr2_writes,
        );
        reg2
    };
    let (reg3, _) = tester.write_heap_pointer_default(RV64_REGISTER_NUM_LIMBS, pointer_increment);

    rv64_heap_instruction(reg3, reg1, reg2, opcode_with_offset)
}

fn rv64_heap_instruction(
    rd: usize,
    rs1: usize,
    rs2: usize,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    Instruction::from_isize(
        VmOpcode::from_usize(opcode_with_offset),
        rd as isize,
        rs1 as isize,
        rs2 as isize,
        1_isize,
        2_isize,
    )
}

fn write_u16_heap_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    reg_increment: usize,
    pointer_increment: usize,
    writes: Vec<[BabyBear; NUM_LIMBS]>,
) -> (usize, usize) {
    const { assert!(NUM_LIMBS.is_multiple_of(BLOCK_FE_WIDTH)) };

    let register = tester.get_default_register(reg_increment);
    let pointer = tester.get_default_pointer(pointer_increment);
    write_ptr_reg(tester, 1, register, pointer as u64);

    for (i, &write) in writes.iter().enumerate() {
        let byte_ptr = pointer + i * NUM_LIMBS * U16_CELL_SIZE;
        for j in (0..NUM_LIMBS).step_by(BLOCK_FE_WIDTH) {
            let cell_idx = byte_ptr / U16_CELL_SIZE + j;
            tester.write::<BLOCK_FE_WIDTH>(
                2usize,
                cell_idx,
                write[j..j + BLOCK_FE_WIDTH].try_into().unwrap(),
            );
        }
    }

    (register, pointer)
}

pub fn rv64_heap_branch_default<const NUM_LIMBS: usize>(
    tester: &mut impl TestBuilder<BabyBear>,
    addr1_writes: Vec<[BabyBear; NUM_LIMBS]>,
    addr2_writes: Vec<[BabyBear; NUM_LIMBS]>,
    imm: isize,
    opcode_with_offset: usize,
) -> Instruction<BabyBear> {
    let (reg1, _) =
        tester.write_heap_default::<NUM_LIMBS>(RV64_REGISTER_NUM_LIMBS, 128, addr1_writes);
    let reg2 = if addr2_writes.is_empty() {
        0
    } else {
        let (reg2, _) =
            tester.write_heap_default::<NUM_LIMBS>(RV64_REGISTER_NUM_LIMBS, 128, addr2_writes);
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
pub fn rv64_rand_write_register_or_imm<const NUM_LIMBS: usize>(
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

    tester.write::<NUM_LIMBS>(1, rs1, rs1_writes.map(BabyBear::from_u32));
    if !rs2_is_imm {
        tester.write::<NUM_LIMBS>(1, rs2, rs2_writes.map(BabyBear::from_u32));
    }

    (
        Instruction::from_usize(
            VmOpcode::from_usize(opcode_with_offset),
            [rd, rs1, rs2, 1, if rs2_is_imm { 0 } else { 1 }],
        ),
        rd,
    )
}

pub fn generate_rv64_is_type_immediate(
    rng: &mut StdRng,
) -> (usize, [u32; RV64_REGISTER_NUM_LIMBS]) {
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
        ]
        .map(|x| x as u32),
    )
}
