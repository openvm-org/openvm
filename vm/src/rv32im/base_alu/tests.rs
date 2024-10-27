use std::{borrow::BorrowMut, sync::Arc};

use ax_circuit_primitives::xor::XorLookupChip;
use ax_stark_sdk::utils::create_seeded_rng;
use ax_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, ChipUsageGetter,
};
use axvm_instructions::instruction::Instruction;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use rand::{rngs::StdRng, Rng};

use super::{core::run_alu, BaseAluCoreChip, Rv32BaseAluChip};
use crate::{
    arch::{
        instructions::AluOpcode,
        testing::{memory::gen_pointer, TestAdapterChip, VmChipTestBuilder},
        ExecutionBridge, InstructionExecutor, VmAdapterChip, VmChipWrapper, BYTE_XOR_BUS,
    },
    rv32im::{
        adapters::{Rv32BaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        base_alu::BaseAluCoreCols,
    },
    utils::{generate_long_number, generate_rv32_is_type_immediate},
};

type F = BabyBear;

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_rv32_alu_rand_write_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: AluOpcode,
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    c_imm: Option<usize>,
    rng: &mut StdRng,
) {
    let is_imm = c_imm.is_some();

    let rs1 = gen_pointer(rng, 4);
    let rs2 = c_imm.unwrap_or_else(|| gen_pointer(rng, 4));
    let rd = gen_pointer(rng, 4);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    if !is_imm {
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));
    }

    let a = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [rd, rs1, rs2, 1, if is_imm { 0 } else { 1 }],
        ),
    );

    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    );
}

fn run_rv32_alu_rand_test(opcode: AluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32BaseAluChip::<F>::new(
        Rv32BaseAluAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        BaseAluCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let (c_imm, c) = if rng.gen_bool(0.5) {
            (
                None,
                generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng),
            )
        } else {
            let (imm, c) = generate_rv32_is_type_immediate(&mut rng);
            (Some(imm), c)
        };
        run_rv32_alu_rand_write_execute(&mut tester, &mut chip, opcode, b, c, c_imm, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_alu_add_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::ADD, 100);
}

#[test]
fn rv32_alu_sub_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::SUB, 100);
}

#[test]
fn rv32_alu_xor_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::XOR, 100);
}

#[test]
fn rv32_alu_or_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::OR, 100);
}

#[test]
fn rv32_alu_and_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::AND, 100);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adapter is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

type Rv32BaseAluTestChip<F> =
    VmChipWrapper<F, TestAdapterChip<F>, BaseAluCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

#[allow(clippy::too_many_arguments)]
fn run_rv32_alu_negative_test(
    opcode: AluOpcode,
    a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    interaction_error: bool,
) {
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut chip = Rv32BaseAluTestChip::<F>::new(
        TestAdapterChip::new(
            vec![[b.map(F::from_canonical_u32), c.map(F::from_canonical_u32)].concat()],
            vec![None],
            ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
        ),
        BaseAluCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    tester.execute(
        &mut chip,
        Instruction::from_usize(opcode as usize, [0, 0, 0, 1, 1]),
    );

    let trace_width = chip.trace_width();
    let adapter_width = BaseAir::<F>::width(chip.adapter.air());

    if (opcode == AluOpcode::ADD || opcode == AluOpcode::SUB)
        && a.iter().all(|&a_val| a_val < (1 << RV32_CELL_BITS))
    {
        xor_lookup_chip.clear();
        for a_val in a {
            xor_lookup_chip.request(a_val, a_val);
        }
    }

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BaseAluCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = a.map(F::from_canonical_u32);
        *trace = RowMajorMatrix::new(values, trace_width);
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(xor_lookup_chip)
        .finalize();
    tester.simple_test_with_expected_error(if interaction_error {
        VerificationError::NonZeroCumulativeSum
    } else {
        VerificationError::OodEvaluationMismatch
    });
}

#[test]
fn rv32_alu_add_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::ADD,
        [246, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        false,
    );
}

#[test]
fn rv32_alu_add_out_of_range_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::ADD,
        [500, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        true,
    );
}

#[test]
fn rv32_alu_sub_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::SUB,
        [255, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        false,
    );
}

#[test]
fn rv32_alu_sub_out_of_range_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::SUB,
        [F::neg_one().as_canonical_u32(), 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        true,
    );
}

#[test]
fn rv32_alu_xor_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::XOR,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [255, 255, 255, 255],
        true,
    );
}

#[test]
fn rv32_alu_or_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::OR,
        [255, 255, 255, 255],
        [255, 255, 255, 254],
        [0, 0, 0, 0],
        true,
    );
}

#[test]
fn rv32_alu_and_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::AND,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [23, 205, 73, 49];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AluOpcode::ADD, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_sub_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [179, 118, 240, 172];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AluOpcode::SUB, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_xor_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [215, 138, 49, 173];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AluOpcode::XOR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_or_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [247, 171, 61, 239];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AluOpcode::OR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_and_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [32, 33, 12, 66];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AluOpcode::AND, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
