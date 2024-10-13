use std::array;

use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{Rv32JalrChip, Rv32JalrCoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32JalrOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        Rv32JalrAdapter, PC_BITS,
    },
    program::Instruction,
    rv32_jalr::solve_jalr,
};

const IMM_BITS: usize = 12;
type F = BabyBear;

fn into_limbs(num: u32) -> [F; 4] {
    array::from_fn(|i| F::from_canonical_u32((num >> (8 * i)) & 255))
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32JalrChip<F>,
    rng: &mut StdRng,
    opcode: Rv32JalrOpcode,
) {
    let imm: i32 = rng.gen_range(0..(1 << IMM_BITS)) - (1 << (IMM_BITS - 1));
    let a = gen_pointer(rng, 32);
    let b = gen_pointer(rng, 32);
    let rs1 = if imm < 0 {
        rng.gen_range(-imm..(1 << PC_BITS))
    } else {
        rng.gen_range(0..((1 << PC_BITS) - imm))
    } as u32;
    let rs1_limbs = into_limbs(rs1);
    tester.write(1, b, rs1_limbs);

    tester.execute(
        chip,
        Instruction::from_isize(
            opcode as usize + Rv32JalrOpcode::default_offset(),
            a as isize,
            b as isize,
            imm as isize,
            1,
            0,
        ),
    );
    let initial_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .initial_state
        .pc
        .as_canonical_u32();
    let final_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .final_state
        .pc
        .as_canonical_u32();

    let (next_pc, rd_data) = solve_jalr(opcode, initial_pc, imm, rs1);

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u32), tester.read::<4>(1, a));
}

#[test]
fn simple_execute_roundtrip_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32JalrAdapter::<F>::new();
    let inner = Rv32JalrCoreChip::<F>::new(Rv32JalrOpcode::default_offset());
    let mut chip = Rv32JalrChip::<F>::new(adapter, inner, tester.memory_chip());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JALR);
    }
}

#[test]
fn solve_jalr_sanity_test() {
    let opcode = JALR;
    let initial_pc = 789456120;
    let imm = -1235;
    let rs1 = 736482910;
    let (next_pc, rd_data) = solve_jalr(opcode, initial_pc, imm, rs1);
    assert_eq!(next_pc, 736481674);
    assert_eq!(rd_data, [252, 36, 14, 47]);
}
