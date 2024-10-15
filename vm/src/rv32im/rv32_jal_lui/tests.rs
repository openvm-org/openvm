use std::sync::Arc;

use afs_primitives::xor::lookup::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{solve_jal_lui, Rv32JalLuiChip, Rv32JalLuiCoreChip};
use crate::{
    arch::{
        instructions::Rv32JalLuiOpcode::{self, *},
        testing::VmChipTestBuilder,
    },
    kernels::core::BYTE_XOR_BUS,
    rv32im::adapters::{Rv32RdWriteAdapter, RV32_CELL_BITS},
    system::program::Instruction,
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

    let a = rng.gen_range(1..32) << 2;

    tester.execute(
        chip,
        Instruction::from_isize(opcode as usize, a as isize, 0, imm as isize, 1, 0),
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

    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u32), tester.read::<4>(1, a));
}

#[test]
fn simple_execute_roundtrip_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));

    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32RdWriteAdapter::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let inner = Rv32JalLuiCoreChip::new(xor_lookup_chip, 0);
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, inner, tester.memory_controller());
    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JAL);
        set_and_execute(&mut tester, &mut chip, &mut rng, LUI);
    }
}

#[test]
fn rand_jal_lui_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));

    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32RdWriteAdapter::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let inner = Rv32JalLuiCoreChip::new(xor_lookup_chip.clone(), 0);
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, inner, tester.memory_controller());
    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JAL);
        set_and_execute(&mut tester, &mut chip, &mut rng, LUI);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
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
