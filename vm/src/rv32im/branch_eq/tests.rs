use std::array;

use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use rand::{rngs::StdRng, Rng};

use super::{
    core::{run_eq, BranchEqualCoreChip},
    Rv32BranchEqualChip,
};
use crate::{
    arch::{
        instructions::{BranchEqualOpcode, UsizeOpcode},
        testing::{memory::gen_pointer, VmChipTestBuilder},
        BasicAdapterInterface, InstructionExecutor, VmCoreChip,
    },
    rv32im::adapters::{
        JumpUiProcessedInstruction, Rv32BranchAdapterChip, PC_BITS, RV32_REGISTER_NUM_LIMBS,
        RV_B_TYPE_IMM_BITS,
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

#[allow(clippy::too_many_arguments)]
fn run_rv32_branch_eq_rand_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: BranchEqualOpcode,
    a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    imm: i32,
    rng: &mut StdRng,
) {
    let rs1 = gen_pointer(rng, 32);
    let rs2 = gen_pointer(rng, 32);
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_canonical_u32));

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

    let (cmp_result, _, _) = run_eq::<F, RV32_REGISTER_NUM_LIMBS>(opcode, &a, &b);
    let from_pc = tester.execution.last_from_pc().as_canonical_u32() as i32;
    let to_pc = tester.execution.last_to_pc().as_canonical_u32() as i32;
    // TODO: update the default increment (i.e. 4) when opcodes are updated
    let pc_inc = if cmp_result { imm } else { 4 };

    assert_eq!(to_pc, from_pc + pc_inc);
}

fn run_rv32_branch_eq_rand_test(opcode: BranchEqualOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    const ABS_MAX_BRANCH: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32BranchEqualChip::<F>::new(
        Rv32BranchAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        BranchEqualCoreChip::new(0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let a = array::from_fn(|_| rng.gen_range(0..F::ORDER_U32));
        let b = if rng.gen_bool(0.5) {
            a
        } else {
            array::from_fn(|_| rng.gen_range(0..F::ORDER_U32))
        };
        let imm = rng.gen_range((-ABS_MAX_BRANCH)..ABS_MAX_BRANCH);
        run_rv32_branch_eq_rand_execute(&mut tester, &mut chip, opcode, a, b, imm, &mut rng);
    }

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_beq_rand_test() {
    run_rv32_branch_eq_rand_test(BranchEqualOpcode::BEQ, 12);
}

#[test]
fn rv32_bne_rand_test() {
    run_rv32_branch_eq_rand_test(BranchEqualOpcode::BNE, 12);
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
    let core = BranchEqualCoreChip::<RV32_REGISTER_NUM_LIMBS>::new(0);

    let mut instruction = Instruction::<F> {
        opcode: BranchEqualOpcode::BEQ.as_usize(),
        c: F::from_canonical_u8(8),
        ..Default::default()
    };
    let x: [F; RV32_REGISTER_NUM_LIMBS] = [19, 4, 1790, 60].map(F::from_canonical_u32);
    let y: [F; RV32_REGISTER_NUM_LIMBS] = [19, 32, 1804, 60].map(F::from_canonical_u32);

    let result = <BranchEqualCoreChip<RV32_REGISTER_NUM_LIMBS> as VmCoreChip<
        F,
        BasicAdapterInterface<F, JumpUiProcessedInstruction<F>, 2, 0, RV32_REGISTER_NUM_LIMBS, 0>,
    >>::execute_instruction(&core, &instruction, 0, [x, y]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_none());

    instruction.opcode = BranchEqualOpcode::BNE.as_usize();
    let result = <BranchEqualCoreChip<RV32_REGISTER_NUM_LIMBS> as VmCoreChip<
        F,
        BasicAdapterInterface<F, JumpUiProcessedInstruction<F>, 2, 0, RV32_REGISTER_NUM_LIMBS, 0>,
    >>::execute_instruction(&core, &instruction, 0, [x, y]);
    let (output, _) = result.expect("execute_instruction failed");
    assert!(output.to_pc.is_some());
    assert_eq!(output.to_pc.unwrap(), 8);
}

#[test]
fn run_eq_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [19, 4, 1790, 60];
    let (cmp_result, _, diff_val) =
        run_eq::<F, RV32_REGISTER_NUM_LIMBS>(BranchEqualOpcode::BEQ, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_val, F::zero());

    let (cmp_result, _, diff_val) =
        run_eq::<F, RV32_REGISTER_NUM_LIMBS>(BranchEqualOpcode::BNE, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_val, F::zero());
}

#[test]
fn run_ne_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [19, 4, 1790, 60];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [19, 32, 1804, 60];
    let (cmp_result, diff_idx, diff_val) =
        run_eq::<F, RV32_REGISTER_NUM_LIMBS>(BranchEqualOpcode::BEQ, &x, &y);
    assert!(!cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u32(x[diff_idx]) - F::from_canonical_u32(y[diff_idx])),
        F::one()
    );

    let (cmp_result, diff_idx, diff_val) =
        run_eq::<F, RV32_REGISTER_NUM_LIMBS>(BranchEqualOpcode::BNE, &x, &y);
    assert!(cmp_result);
    assert_eq!(
        diff_val * (F::from_canonical_u32(x[diff_idx]) - F::from_canonical_u32(y[diff_idx])),
        F::one()
    );
}
