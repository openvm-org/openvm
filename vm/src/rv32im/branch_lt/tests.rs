use std::{array, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{
    core::{solve_cmp, BranchLessThanCoreChip},
    Rv32BranchLessThanChip,
};
use crate::{
    arch::{
        instructions::{BranchLessThanOpcode, UsizeOpcode},
        testing::{memory::gen_pointer, VmChipTestBuilder},
        BasicAdapterInterface, InstructionExecutor, VmCoreChip,
    },
    kernels::core::BYTE_XOR_BUS,
    rv32im::adapters::{
        JumpUiProcessedInstruction, Rv32BranchAdapterChip, PC_BITS, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LANES, RV_B_TYPE_IMM_BITS,
    },
    system::program::Instruction,
};

type F = BabyBear;

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|_| rng.gen_range(0..(1 << LIMB_BITS)))
}

#[allow(clippy::too_many_arguments)]
fn run_rv32_branch_lt_rand_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: BranchLessThanOpcode,
    a: [u32; RV32_REGISTER_NUM_LANES],
    b: [u32; RV32_REGISTER_NUM_LANES],
    imm: i32,
    rng: &mut StdRng,
) {
    let rs1 = gen_pointer(rng, 32);
    let rs2 = gen_pointer(rng, 32);
    tester.write::<RV32_REGISTER_NUM_LANES>(1, rs1, a.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LANES>(1, rs2, b.map(F::from_canonical_u32));

    tester.execute_with_pc(
        chip,
        Instruction::from_isize(
            opcode as usize,
            rs1 as isize,
            rs2 as isize,
            imm as isize,
            1,
            1,
        ),
        rng.gen_range(imm.unsigned_abs()..(1 << PC_BITS)),
    );

    let (cmp_result, _, _, _) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(opcode, &a, &b);
    let from_pc = tester.execution.last_from_pc().as_canonical_u32() as i32;
    let to_pc = tester.execution.last_to_pc().as_canonical_u32() as i32;
    // TODO: update the default increment (i.e. 4) when opcodes are updated
    let pc_inc = if cmp_result { imm } else { 4 };

    assert_eq!(to_pc, from_pc + pc_inc);
}

fn run_rv32_branch_lt_rand_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32BranchLessThanChip::<F>::new(
        Rv32BranchAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        BranchLessThanCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let a = generate_long_number::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(&mut rng);
        let b = if rng.gen_bool(0.5) {
            a
        } else {
            generate_long_number::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(&mut rng)
        };
        let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        run_rv32_branch_lt_rand_execute(&mut tester, &mut chip, opcode, a, b, imm, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_blt_rand_test() {
    run_rv32_branch_lt_rand_test(BranchLessThanOpcode::BLT, 10);
}

#[test]
fn rv32_bltu_rand_test() {
    run_rv32_branch_lt_rand_test(BranchLessThanOpcode::BLTU, 12);
}

#[test]
fn rv32_bge_rand_test() {
    run_rv32_branch_lt_rand_test(BranchLessThanOpcode::BGE, 12);
}

#[test]
fn rv32_bgeu_rand_test() {
    run_rv32_branch_lt_rand_test(BranchLessThanOpcode::BGEU, 12);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adapter is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

// TODO: write negative tests

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn execute_pc_increment_sanity_test() {
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let core =
        BranchLessThanCoreChip::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>::new(xor_lookup_chip, 0);

    let mut instruction = Instruction::<F> {
        opcode: BranchLessThanOpcode::BLT.as_usize(),
        op_c: F::from_canonical_u8(8),
        ..Default::default()
    };
    let x: [F; RV32_REGISTER_NUM_LANES] = [145, 34, 25, 205].map(F::from_canonical_u32);

    let result = <BranchLessThanCoreChip<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS> as VmCoreChip<
        F,
        BasicAdapterInterface<F, JumpUiProcessedInstruction<F>, 2, 0, RV32_REGISTER_NUM_LANES, 0>,
    >>::execute_instruction(&core, &instruction, 0, [x, x]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_none());

    instruction.opcode = BranchLessThanOpcode::BGE.as_usize();
    let result = <BranchLessThanCoreChip<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS> as VmCoreChip<
        F,
        BasicAdapterInterface<F, JumpUiProcessedInstruction<F>, 2, 0, RV32_REGISTER_NUM_LANES, 0>,
    >>::execute_instruction(&core, &instruction, 0, [x, x]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_some());
    assert_eq!(output.to_pc.unwrap(), 8);
}

#[test]
fn solve_cmp_unsigned_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [145, 34, 25, 205];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BLTU, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BGEU, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn solve_cmp_same_sign_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [145, 34, 25, 205];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BLT, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BGE, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn solve_cmp_diff_sign_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LANES] = [45, 35, 25, 55];
    let y: [u32; RV32_REGISTER_NUM_LANES] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BLT, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        solve_cmp::<RV32_REGISTER_NUM_LANES, RV32_CELL_BITS>(BranchLessThanOpcode::BGE, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}
