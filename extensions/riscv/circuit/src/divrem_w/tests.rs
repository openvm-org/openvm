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
use openvm_riscv_transpiler::DivRemWOpcode::{self, *};
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

use super::{DivRemWCoreAir, DivRemWFiller, Rv64DivRemWChip};
use crate::{
    adapters::{
        Rv64MultWAdapterAir, Rv64MultWAdapterCols, Rv64MultWAdapterExecutor,
        Rv64MultWAdapterFiller, RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
    },
    divrem::{run_mul_carries, run_sltu_diff_idx, DivRemCoreCols, DivRemCoreSpecialCase},
    test_utils::get_verification_error,
    Rv64DivRemWAir, Rv64DivRemWExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1 << RV64_CELL_BITS) as u32,
    (MAX_NUM_LIMBS * (1 << RV64_CELL_BITS)),
];
type Harness = TestChipHarness<F, Rv64DivRemWExecutor, Rv64DivRemWAir, Rv64DivRemWChip<F>>;
type DivRemWCoreCols<T> = DivRemCoreCols<T, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

fn limb_sra<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: [u32; NUM_LIMBS],
    shift: usize,
) -> [u32; NUM_LIMBS] {
    assert!(shift < NUM_LIMBS);
    let ext = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
    array::from_fn(|i| if i + shift < NUM_LIMBS { x[i] } else { ext })
}

fn word_to_register(x: [u32; RV64_WORD_NUM_LIMBS]) -> [u32; RV64_REGISTER_NUM_LIMBS] {
    let mut out = [0; RV64_REGISTER_NUM_LIMBS];
    out[..RV64_WORD_NUM_LIMBS].copy_from_slice(&x);
    out
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64DivRemWAir, Rv64DivRemWExecutor, Rv64DivRemWChip<F>) {
    let air = Rv64DivRemWAir::new(
        Rv64MultWAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        DivRemWCoreAir::new(
            bitwise_chip.bus(),
            *range_tuple_chip.bus(),
            DivRemWOpcode::CLASS_OFFSET,
        ),
    );
    let executor = Rv64DivRemWExecutor::new(Rv64MultWAdapterExecutor, DivRemWOpcode::CLASS_OFFSET);
    let chip = Rv64DivRemWChip::<F>::new(
        DivRemWFiller::new(
            Rv64MultWAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            range_tuple_chip,
            DivRemWOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
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

/// Execute a divrem_w instruction over full 8-byte register inputs.
/// Only the low 32-bit word participates in the core div/rem relation; upper bytes are
/// consumed by adapter full-width read constraints.
#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: DivRemWOpcode,
    b: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or_else(|| {
        let mut b = [0; RV64_REGISTER_NUM_LIMBS];
        let b_word = generate_long_number::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(rng);
        b[..RV64_WORD_NUM_LIMBS].copy_from_slice(&b_word);
        for b_high in b[RV64_WORD_NUM_LIMBS..].iter_mut() {
            *b_high = rng.gen_range(0..256);
        }
        b
    });
    let c = c.unwrap_or_else(|| {
        let mut c = [0; RV64_REGISTER_NUM_LIMBS];
        let c_word = limb_sra::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(
            generate_long_number::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(rng),
            rng.gen_range(0..(RV64_WORD_NUM_LIMBS - 1)),
        );
        c[..RV64_WORD_NUM_LIMBS].copy_from_slice(&c_word);
        for c_high in c[RV64_WORD_NUM_LIMBS..].iter_mut() {
            *c_high = rng.gen_range(0..256);
        }
        c
    });

    // Write full 8-byte registers. Upper bytes are arbitrary and remain adapter-constrained.
    let rs1_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rd_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    tester.write::<RV64_REGISTER_NUM_LIMBS>(1, rs1_ptr, b.map(F::from_canonical_u32));
    tester.write::<RV64_REGISTER_NUM_LIMBS>(1, rs2_ptr, c.map(F::from_canonical_u32));

    let b_word: [u32; RV64_WORD_NUM_LIMBS] = b[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let c_word: [u32; RV64_WORD_NUM_LIMBS] = c[..RV64_WORD_NUM_LIMBS].try_into().unwrap();

    let is_div = opcode == DIVW || opcode == DIVUW;
    let is_signed = opcode == DIVW || opcode == REMW;

    let (q, r, _, _, _, _) = crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(
        is_signed, &b_word, &c_word,
    );

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd_ptr, rs1_ptr, rs2_ptr, 1, 0]),
    );

    // The core result is sign-extended from 32-bit to 64-bit by the adapter.
    let core_result = if is_div { q } else { r };
    let result_word: [u8; RV64_WORD_NUM_LIMBS] = core_result.map(|x| x as u8);
    let sign_extend_byte = ((1u16 << RV64_CELL_BITS) - 1) as u8
        * (result_word[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS as u8 - 1));
    let mut expected = [sign_extend_byte; RV64_REGISTER_NUM_LIMBS];
    for i in 0..RV64_WORD_NUM_LIMBS {
        expected[i] = result_word[i];
    }

    assert_eq!(
        expected.map(F::from_canonical_u8),
        tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd_ptr)
    );
}

// Test special cases in addition to random cases (i.e. zero divisor with b > 0,
// zero divisor with b < 0, r = 0 (3 cases), and signed overflow).
fn set_and_execute_special_cases<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: DivRemWOpcode,
) {
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([98, 188, 163, 127])),
        Some(word_to_register([0, 0, 0, 0])),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([98, 188, 163, 229])),
        Some(word_to_register([0, 0, 0, 0])),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([0, 0, 0, 128])),
        Some(word_to_register([0, 1, 0, 0])),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([0, 0, 0, 127])),
        Some(word_to_register([0, 1, 0, 0])),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([0, 0, 0, 0])),
        Some(word_to_register([0, 0, 0, 0])),
    );
    set_and_execute(
        tester,
        executor,
        arena,
        rng,
        opcode,
        Some(word_to_register([0, 0, 0, 128])),
        Some(word_to_register([255, 255, 255, 255])),
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(DIVW, 100)]
#[test_case(DIVUW, 100)]
#[test_case(REMW, 100)]
#[test_case(REMUW, 100)]
fn rand_divremw_test(opcode: DivRemWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&tester);

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
struct DivRemWPrankValues {
    pub q: Option<[u32; RV64_WORD_NUM_LIMBS]>,
    pub r: Option<[u32; RV64_WORD_NUM_LIMBS]>,
    pub r_prime: Option<[u32; RV64_WORD_NUM_LIMBS]>,
    pub diff_val: Option<u32>,
    pub zero_divisor: Option<bool>,
    pub r_zero: Option<bool>,
    pub rs1: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    pub rs2: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    pub result_sign: Option<u32>,
}

fn run_negative_divremw_test(
    opcode: DivRemWOpcode,
    b: [u32; RV64_REGISTER_NUM_LIMBS],
    c: [u32; RV64_REGISTER_NUM_LIMBS],
    prank_vals: DivRemWPrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let b_word: [u32; RV64_WORD_NUM_LIMBS] = b[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let c_word: [u32; RV64_WORD_NUM_LIMBS] = c[..RV64_WORD_NUM_LIMBS].try_into().unwrap();

    let is_div = opcode == DIVW || opcode == DIVUW;
    let is_signed = opcode == DIVW || opcode == REMW;
    let (expected_q, expected_r, _, _, _, _) = crate::divrem::run_divrem::<
        RV64_WORD_NUM_LIMBS,
        RV64_CELL_BITS,
    >(is_signed, &b_word, &c_word);
    let default_result_sign = prank_vals.result_sign.unwrap_or_else(|| {
        let output_word = if is_div {
            prank_vals.q.unwrap_or(expected_q)
        } else {
            prank_vals.r.unwrap_or(expected_r)
        };
        (output_word[RV64_WORD_NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1)) & 1
    });

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64MultWAdapterCols<F> = adapter_row.borrow_mut();
        let cols: &mut DivRemWCoreCols<F> = core_row.borrow_mut();

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
        if let Some(rs1) = prank_vals.rs1 {
            let rs1_word: [u32; RV64_WORD_NUM_LIMBS] =
                rs1[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
            let rs1_high: [u32; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS] =
                rs1[RV64_WORD_NUM_LIMBS..].try_into().unwrap();
            cols.b = rs1_word.map(F::from_canonical_u32);
            adapter_cols.rs1_high = rs1_high.map(F::from_canonical_u32);
        }
        if let Some(rs2) = prank_vals.rs2 {
            let rs2_word: [u32; RV64_WORD_NUM_LIMBS] =
                rs2[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
            let rs2_high: [u32; RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS] =
                rs2[RV64_WORD_NUM_LIMBS..].try_into().unwrap();
            cols.c = rs2_word.map(F::from_canonical_u32);
            adapter_cols.rs2_high = rs2_high.map(F::from_canonical_u32);
        }
        adapter_cols.result_sign = F::from_canonical_u32(default_result_sign);

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
fn rv64_divremw_unsigned_wrong_q_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([98, 188, 163, 229]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([123, 34, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([245, 168, 7, 0]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
    run_negative_divremw_test(REMUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_unsigned_wrong_r_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([98, 188, 163, 229]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([123, 34, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        r: Some([171, 3, 0, 0]),
        r_prime: Some([171, 3, 0, 0]),
        diff_val: Some(31),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
    run_negative_divremw_test(REMUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_unsigned_high_mult_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 2, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([128, 0, 0, 1]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
    run_negative_divremw_test(REMUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_unsigned_zero_divisor_wrong_r_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([254, 255, 255, 255]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        r: Some([255, 255, 255, 255]),
        r_prime: Some([255, 255, 255, 255]),
        diff_val: Some(255),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
    run_negative_divremw_test(REMUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_signed_wrong_q_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([98, 188, 163, 229]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([123, 34, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([74, 61, 255, 255]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, true);
    run_negative_divremw_test(REMW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_signed_wrong_r_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([98, 188, 163, 229]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([123, 34, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        r: Some([212, 241, 255, 255]),
        r_prime: Some([44, 14, 0, 0]),
        diff_val: Some(20),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, true);
    run_negative_divremw_test(REMW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_signed_high_mult_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 255]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 255]);
    let prank_vals = DivRemWPrankValues {
        q: Some([1, 0, 0, 1]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, true);
    run_negative_divremw_test(REMW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_signed_r_wrong_sign_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([50, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([31, 5, 0, 0]),
        r: Some([242, 255, 255, 255]),
        r_prime: Some([242, 255, 255, 255]),
        diff_val: Some(192),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_signed_r_wrong_prime_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([50, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([31, 5, 0, 0]),
        r: Some([242, 255, 255, 255]),
        r_prime: Some([14, 0, 0, 0]),
        diff_val: Some(36),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_signed_zero_divisor_wrong_r_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([254, 255, 255, 255]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        r: Some([255, 255, 255, 255]),
        r_prime: Some([1, 0, 0, 0]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, true);
    run_negative_divremw_test(REMW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_false_zero_divisor_flag_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([50, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([29, 5, 0, 0]),
        r: Some([86, 0, 0, 0]),
        r_prime: Some([86, 0, 0, 0]),
        diff_val: Some(36),
        zero_divisor: Some(true),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, false);
    run_negative_divremw_test(REMUW, b, c, prank_vals, false);
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_false_r_zero_flag_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([50, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        q: Some([29, 5, 0, 0]),
        r: Some([86, 0, 0, 0]),
        r_prime: Some([86, 0, 0, 0]),
        diff_val: Some(36),
        r_zero: Some(true),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, false);
    run_negative_divremw_test(REMUW, b, c, prank_vals, false);
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_unset_zero_divisor_flag_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        zero_divisor: Some(false),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, false);
    run_negative_divremw_test(REMUW, b, c, prank_vals, false);
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_wrong_r_zero_flag_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        zero_divisor: Some(false),
        r_zero: Some(true),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, false);
    run_negative_divremw_test(REMUW, b, c, prank_vals, false);
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_unset_r_zero_flag_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([0, 0, 1, 0]);
    let prank_vals = DivRemWPrankValues {
        r_zero: Some(false),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, false);
    run_negative_divremw_test(REMUW, b, c, prank_vals, false);
    run_negative_divremw_test(DIVW, b, c, prank_vals, false);
    run_negative_divremw_test(REMW, b, c, prank_vals, false);
}

#[test]
fn rv64_divremw_adapter_wrong_rs1_upper_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([7, 0, 0, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([2, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        rs1: Some([7, 0, 0, 0, 1, 2, 3, 4]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_adapter_wrong_rs2_upper_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([7, 0, 0, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([2, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        rs2: Some([2, 0, 0, 0, 5, 6, 7, 8]),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_wrong_upper_sign_extension_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([7, 0, 0, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([2, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        result_sign: Some(1),
        ..Default::default()
    };
    run_negative_divremw_test(DIVUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_wrong_upper_sign_extension_negative_to_zero_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([254, 255, 255, 255]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([1, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        result_sign: Some(0),
        ..Default::default()
    };
    run_negative_divremw_test(DIVW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_wrong_upper_sign_extension_remuw_negative_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([7, 0, 0, 0]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([2, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        result_sign: Some(1),
        ..Default::default()
    };
    run_negative_divremw_test(REMUW, b, c, prank_vals, true);
}

#[test]
fn rv64_divremw_wrong_upper_sign_extension_remw_negative_to_zero_test() {
    let b: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([254, 255, 255, 255]);
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = word_to_register([3, 0, 0, 0]);
    let prank_vals = DivRemWPrankValues {
        result_sign: Some(0),
        ..Default::default()
    };
    run_negative_divremw_test(REMW, b, c, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_divremw_unsigned_sanity_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [245, 168, 6, 0];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [171, 4, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(false, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divremw_unsigned_zero_divisor_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(false, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::ZeroDivisor);
}

#[test]
fn run_divremw_signed_sanity_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [74, 60, 255, 255];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [212, 240, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert!(q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divremw_signed_zero_divisor_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert!(q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::ZeroDivisor);
}

#[test]
fn run_divremw_signed_overflow_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [255, 255, 255, 255];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(x[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::SignedOverflow);
}

#[test]
fn run_divremw_signed_min_dividend_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [123, 34, 255, 255];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [236, 147, 0, 0];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [156, 149, 255, 255];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert!(!q_sign);
    assert_eq!(case, DivRemCoreSpecialCase::None);
}

#[test]
fn run_divremw_zero_quotient_test() {
    let x: [u32; RV64_WORD_NUM_LIMBS] = [255, 255, 255, 255];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 1];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign, q_sign, case) =
        crate::divrem::run_divrem::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    for i in 0..RV64_WORD_NUM_LIMBS {
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
    let x: [u32; RV64_WORD_NUM_LIMBS] = [123, 34, 254, 67];
    let y: [u32; RV64_WORD_NUM_LIMBS] = [123, 34, 255, 67];
    assert_eq!(run_sltu_diff_idx(&x, &y, true), 2);
    assert_eq!(run_sltu_diff_idx(&y, &x, false), 2);
    assert_eq!(run_sltu_diff_idx(&x, &x, false), RV64_WORD_NUM_LIMBS);
}

#[test]
fn run_mul_carries_signed_sanity_test() {
    let d: [u32; RV64_WORD_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [200, 8, 68, 255];
    let c = [40, 101, 126, 206, 304, 376, 450, 464];
    let carry = run_mul_carries::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(true, &d, &q, &r, true);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}

#[test]
fn run_mul_unsigned_sanity_test() {
    let d: [u32; RV64_WORD_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV64_WORD_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV64_WORD_NUM_LIMBS] = [200, 8, 68, 255];
    let c = [40, 101, 126, 206, 107, 93, 18, 0];
    let carry = run_mul_carries::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(false, &d, &q, &r, true);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}

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
