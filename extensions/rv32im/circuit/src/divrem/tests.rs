use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS, RANGE_TUPLE_CHECKER_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::generate_long_number,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::DivRemOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32MultAdapterRecord, DivRemCoreRecord, Rv32DivRemChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::core::run_divrem;
use crate::{
    adapters::{
        Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    divrem::{
        run_mul_carries, run_sltu_diff_idx, DivRemCoreCols, DivRemCoreSpecialCase, Rv32DivRemChip,
    },
    test_utils::get_verification_error,
    DivRemCoreAir, DivRemFiller, Rv32DivRemAir, Rv32DivRemExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1 << RV32_CELL_BITS) as u32,
    (MAX_NUM_LIMBS * (1 << RV32_CELL_BITS)),
];
type Harness = TestChipHarness<F, Rv32DivRemExecutor, Rv32DivRemAir, Rv32DivRemChip<F>>;

fn limb_sra<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: [u32; NUM_LIMBS],
    shift: usize,
) -> [u32; NUM_LIMBS] {
    assert!(shift < NUM_LIMBS);
    let ext = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
    array::from_fn(|i| if i + shift < NUM_LIMBS { x[i] } else { ext })
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv32DivRemAir, Rv32DivRemExecutor, Rv32DivRemChip<F>) {
    let air = Rv32DivRemAir::new(
        Rv32MultAdapterAir::new(execution_bridge, memory_bridge),
        DivRemCoreAir::new(
            bitwise_chip.bus(),
            *range_tuple_chip.bus(),
            DivRemOpcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv32DivRemExecutor::new(Rv32MultAdapterExecutor, DivRemOpcode::CLASS_OFFSET);
    let chip = Rv32DivRemChip::<F>::new(
        DivRemFiller::new(
            Rv32MultAdapterFiller,
            bitwise_chip,
            range_tuple_chip,
            DivRemOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_tuple_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, cpu_chip, MAX_INS_CAPACITY);

    (
        harness,
        (bitwise_chip.air, bitwise_chip),
        (range_tuple_chip.air, range_tuple_chip),
    )
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: DivRemOpcode,
    b: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    c: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(generate_long_number::<
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >(rng));
    let c = c.unwrap_or(limb_sra::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(rng),
        rng.gen_range(0..(RV32_REGISTER_NUM_LIMBS - 1)),
    ));

    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    let rd = gen_pointer(rng, 4);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

    let is_div = opcode == DIV || opcode == DIVU;
    let is_signed = opcode == DIV || opcode == REM;

    let (q, r, _, _, _, _) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(is_signed, &b, &c);
    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
    );

    assert_eq!(
        (if is_div { q } else { r }).map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    );
}

// Test special cases in addition to random cases (i.e. zero divisor with b > 0,
// zero divisor with b < 0, r = 0 (3 cases), and signed overflow).
fn set_and_execute_special_cases<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: DivRemOpcode,
) {
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([98, 188, 163, 127]),
        Some([0, 0, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([98, 188, 163, 229]),
        Some([0, 0, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([0, 0, 0, 128]),
        Some([0, 1, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([0, 0, 0, 127]),
        Some([0, 1, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([0, 0, 0, 0]),
        Some([0, 0, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([0, 0, 0, 0]),
        Some([0, 0, 0, 0]),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some([0, 0, 0, 128]),
        Some([255, 255, 255, 255]),
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(DIV, 100)]
#[test_case(DIVU, 100)]
#[test_case(REM, 100)]
#[test_case(REMU, 100)]
fn rand_divrem_test(opcode: DivRemOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }
    set_and_execute_special_cases(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Default, Clone, Copy)]
struct DivRemPrankValues<const NUM_LIMBS: usize> {
    pub q: Option<[u32; NUM_LIMBS]>,
    pub r: Option<[u32; NUM_LIMBS]>,
    pub r_prime: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
    pub zero_divisor: Option<bool>,
    pub r_zero: Option<bool>,
}

fn run_negative_divrem_test(
    opcode: DivRemOpcode,
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    prank_vals: DivRemPrankValues<RV32_REGISTER_NUM_LIMBS>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut DivRemCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(q) = prank_vals.q {
            cols.q = q.map(F::from_canonical_u32);
        }
        if let Some(r) = prank_vals.r {
            cols.r = r.map(F::from_canonical_u32);
            let r_sum = r.iter().sum::<u32>();
            cols.r_sum_inv = F::from_canonical_u32(r_sum)
                .try_inverse()
                .unwrap_or(F::ZERO);
        }
        if let Some(r_prime) = prank_vals.r_prime {
            cols.r_prime = r_prime.map(F::from_canonical_u32);
            cols.r_inv = cols
                .r_prime
                .map(|r| (r - F::from_canonical_u32(256)).inverse());
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.lt_diff = F::from_canonical_u32(diff_val);
        }
        if let Some(zero_divisor) = prank_vals.zero_divisor {
            cols.zero_divisor = F::from_bool(zero_divisor);
        }
        if let Some(r_zero) = prank_vals.r_zero {
            cols.r_zero = F::from_bool(r_zero);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_divrem_unsigned_wrong_q_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([245, 168, 7, 0]),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, true);
    run_negative_divrem_test(REMU, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_unsigned_wrong_r_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let prank_vals = DivRemPrankValues {
        r: Some([171, 3, 0, 0]),
        r_prime: Some([171, 3, 0, 0]),
        diff_val: Some(31),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, true);
    run_negative_divrem_test(REMU, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_unsigned_high_mult_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 2, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([128, 0, 0, 1]),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, true);
    run_negative_divrem_test(REMU, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_unsigned_zero_divisor_wrong_r_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [254, 255, 255, 255];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        r: Some([255, 255, 255, 255]),
        r_prime: Some([255, 255, 255, 255]),
        diff_val: Some(255),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, true);
    run_negative_divrem_test(REMU, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_signed_wrong_q_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([74, 61, 255, 255]),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, true);
    run_negative_divrem_test(REM, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_signed_wrong_r_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let prank_vals = DivRemPrankValues {
        r: Some([212, 241, 255, 255]),
        r_prime: Some([44, 14, 0, 0]),
        diff_val: Some(20),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, true);
    run_negative_divrem_test(REM, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_signed_high_mult_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 255];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 255];
    let prank_vals = DivRemPrankValues {
        q: Some([1, 0, 0, 1]),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, true);
    run_negative_divrem_test(REM, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_signed_r_wrong_sign_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([31, 5, 0, 0]),
        r: Some([242, 255, 255, 255]),
        r_prime: Some([242, 255, 255, 255]),
        diff_val: Some(192),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_signed_r_wrong_prime_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([31, 5, 0, 0]),
        r: Some([242, 255, 255, 255]),
        r_prime: Some([14, 0, 0, 0]),
        diff_val: Some(36),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_signed_zero_divisor_wrong_r_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [254, 255, 255, 255];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        r: Some([255, 255, 255, 255]),
        r_prime: Some([1, 0, 0, 0]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_divrem_test(DIV, b, c, prank_vals, true);
    run_negative_divrem_test(REM, b, c, prank_vals, true);
}

#[test]
fn rv32_divrem_false_zero_divisor_flag_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([29, 5, 0, 0]),
        r: Some([86, 0, 0, 0]),
        r_prime: Some([86, 0, 0, 0]),
        diff_val: Some(36),
        zero_divisor: Some(true),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, false);
    run_negative_divrem_test(REMU, b, c, prank_vals, false);
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_false_r_zero_flag_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [50, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        q: Some([29, 5, 0, 0]),
        r: Some([86, 0, 0, 0]),
        r_prime: Some([86, 0, 0, 0]),
        diff_val: Some(36),
        r_zero: Some(true),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, false);
    run_negative_divrem_test(REMU, b, c, prank_vals, false);
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_unset_zero_divisor_flag_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        zero_divisor: Some(false),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, false);
    run_negative_divrem_test(REMU, b, c, prank_vals, false);
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_wrong_r_zero_flag_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let prank_vals = DivRemPrankValues {
        zero_divisor: Some(false),
        r_zero: Some(true),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, false);
    run_negative_divrem_test(REMU, b, c, prank_vals, false);
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

#[test]
fn rv32_divrem_unset_r_zero_flag_negative_test() {
    let b: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 1, 0];
    let prank_vals = DivRemPrankValues {
        r_zero: Some(false),
        ..Default::default()
    };
    run_negative_divrem_test(DIVU, b, c, prank_vals, false);
    run_negative_divrem_test(REMU, b, c, prank_vals, false);
    run_negative_divrem_test(DIV, b, c, prank_vals, false);
    run_negative_divrem_test(REM, b, c, prank_vals, false);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_divrem_unsigned_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [245, 168, 6, 0];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [171, 4, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divrem_unsigned_zero_divisor_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::ZeroDivisor);
}

#[test]
fn run_divrem_signed_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [74, 60, 255, 255];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [212, 240, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert!(q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divrem_signed_zero_divisor_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert!(q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::ZeroDivisor);
}

#[test]
fn run_divrem_signed_overflow_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(x[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::SignedOverflow);
}

#[test]
fn run_divrem_signed_min_dividend_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 255, 255];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [236, 147, 0, 0];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [156, 149, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divrem_zero_quotient_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 1];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_sltu_diff_idx_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 254, 67];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 255, 67];
    assert_eq!(run_sltu_diff_idx(&x, &y, true), 2);
    assert_eq!(run_sltu_diff_idx(&y, &x, false), 2);
    assert_eq!(run_sltu_diff_idx(&x, &x, false), RV32_REGISTER_NUM_LIMBS);
}

#[test]
fn run_mul_carries_signed_sanity_test() {
    let d: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [200, 8, 68, 255];
    let c = [40, 101, 126, 206, 304, 376, 450, 464];
    let carry = run_mul_carries::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &d, &q, &r, true);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}

#[test]
fn run_mul_unsigned_sanity_test() {
    let d: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [200, 8, 68, 255];
    let c = [40, 101, 126, 206, 107, 93, 18, 0];
    let carry = run_mul_carries::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &d, &q, &r, true);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv32DivRemExecutor, Rv32DivRemAir, Rv32DivRemChipGpu, Rv32DivRemChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_tuple_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32DivRemChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(DIV, 100)]
#[test_case(DIVU, 100)]
#[test_case(REM, 100)]
#[test_case(REMU, 100)]
fn test_cuda_rand_divrem_tracegen(opcode: DivRemOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(default_bitwise_lookup_bus())
        .with_range_tuple_checker(RangeTupleCheckerBus::new(
            RANGE_TUPLE_CHECKER_BUS,
            TUPLE_CHECKER_SIZES,
        ));

    let mut harness = create_cuda_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }
    set_and_execute_special_cases(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        opcode,
    );

    type Record<'a> = (
        &'a mut Rv32MultAdapterRecord,
        &'a mut DivRemCoreRecord<RV32_REGISTER_NUM_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32MultAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
