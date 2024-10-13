use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{solve_jal_lui, Rv32JalLuiChip, Rv32JalLuiIntegration};
use crate::{
    arch::{
        instructions::{
            Rv32JalLuiOpcode::{self, *},
            UsizeOpcode,
        },
        testing::{memory::gen_pointer, VmChipTestBuilder},
        Rv32RdWriteAdapter,
    },
    program::Instruction,
};

const IMM_BITS: usize = 20;

type F = BabyBear;

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32JalLuiChip<F>,
    rng: &mut StdRng,
    opcode: Rv32JalLuiOpcode,
) {
    let imm: i32 = rng.gen_range(0..(1 << IMM_BITS));
    let imm = match opcode {
        JAL => ((imm >> 1) << 2) - (1 << IMM_BITS),
        LUI => imm,
    };

    let a = gen_pointer(rng, 32);

    tester.execute(
        chip,
        Instruction::from_isize(
            opcode as usize + Rv32JalLuiOpcode::default_offset(),
            a as isize,
            0,
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
        .as_canonical_u32() as usize;
    let final_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .final_state
        .pc
        .as_canonical_u32() as usize;

    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u32), tester.read::<4>(1, a));
}

#[test]
fn simple_execute_roundtrip_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32RdWriteAdapter::<F>::new();
    let inner = Rv32JalLuiIntegration::<F>::new(Rv32JalLuiOpcode::default_offset());
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, inner, tester.memory_chip());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JAL);
        set_and_execute(&mut tester, &mut chip, &mut rng, LUI);
    }
}

#[test]
fn solve_jal_sanity_test() {
    let opcode = JAL;
    let initial_pc = 28120;
    let imm = -2048;
    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 26072);
    assert_eq!(rd_data, [220, 109, 0, 0]);
}

#[test]
fn solve_lui_sanity_test() {
    let opcode = LUI;
    let initial_pc = 456789120;
    let imm = 853679;
    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 456789124);
    assert_eq!(rd_data, [0, 240, 106, 208]);
}
