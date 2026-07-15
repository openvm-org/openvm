use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftWOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
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
    crate::{
        adapters::Rv64BaseAluWU16AdapterRecord, Rv64ShiftWLogicalChipGpu,
        Rv64ShiftWRightArithmeticChipGpu, ShiftLogicalCoreRecord, ShiftRightArithmeticCoreRecord,
    },
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    Rv64ShiftWLogicalAir, Rv64ShiftWLogicalChip, Rv64ShiftWLogicalExecutor,
    Rv64ShiftWRightArithmeticAir, Rv64ShiftWRightArithmeticChip, Rv64ShiftWRightArithmeticExecutor,
    ShiftWLogicalCoreAir, ShiftWLogicalFiller, ShiftWRightArithmeticCoreAir,
    ShiftWRightArithmeticFiller,
};
use crate::{
    adapters::{
        Rv64BaseAluWU16AdapterAir, Rv64BaseAluWU16AdapterCols, Rv64BaseAluWU16AdapterExecutor,
        Rv64BaseAluWU16AdapterFiller, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
        RV64_WORD_U16_LIMBS, U16_BITS,
    },
    shift_logical::ShiftLogicalCoreCols,
    shift_right_arithmetic::ShiftRightArithmeticCoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type LogicalHarness =
    TestChipHarness<F, Rv64ShiftWLogicalExecutor, Rv64ShiftWLogicalAir, Rv64ShiftWLogicalChip<F>>;
type RightArithmeticHarness = TestChipHarness<
    F,
    Rv64ShiftWRightArithmeticExecutor,
    Rv64ShiftWRightArithmeticAir,
    Rv64ShiftWRightArithmeticChip<F>,
>;
// SLLW/SRLW/SRAW all use the u16 shift cores over the W adapter.
type ShiftWLogicalCoreCols<T> = ShiftLogicalCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;
type ShiftWRightArithmeticCoreCols<T> =
    ShiftRightArithmeticCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

#[inline(always)]
fn run_shift_w(
    opcode: ShiftWOpcode,
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
) -> ([u8; RV64_REGISTER_NUM_LIMBS], usize, usize) {
    let rs2 = u32::from_le_bytes(*y);
    let (limb_shift, bit_shift) = get_shift_w(y[0]);
    let word_result = match opcode {
        SLLW => (u32::from_le_bytes(*x) << (rs2 & 0x1F)).to_le_bytes(),
        SRLW => (u32::from_le_bytes(*x) >> (rs2 & 0x1F)).to_le_bytes(),
        SRAW => ((i32::from_le_bytes(*x) >> (rs2 & 0x1F)) as u32).to_le_bytes(),
    };
    let sign_extend_limb = ((1u16 << RV64_BYTE_BITS) - 1) as u8
        * (word_result[RV64_WORD_NUM_LIMBS - 1] >> (RV64_BYTE_BITS as u8 - 1));
    let mut result = [sign_extend_limb; RV64_REGISTER_NUM_LIMBS];
    result[..RV64_WORD_NUM_LIMBS].copy_from_slice(&word_result);
    (result, limb_shift, bit_shift)
}

#[inline(always)]
fn get_shift_w(y0: u8) -> (usize, usize) {
    let shift = (y0 as usize) % (RV64_WORD_NUM_LIMBS * RV64_BYTE_BITS);
    (shift / RV64_BYTE_BITS, shift % RV64_BYTE_BITS)
}

fn create_logical_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64ShiftWLogicalAir,
    Rv64ShiftWLogicalExecutor,
    Rv64ShiftWLogicalChip<F>,
) {
    let air = Rv64ShiftWLogicalAir::new(
        Rv64BaseAluWU16AdapterAir::new(execution_bridge, memory_bridge, range_checker_chip.bus()),
        ShiftWLogicalCoreAir::new(range_checker_chip.bus(), ShiftWOpcode::CLASS_OFFSET),
    );
    let executor =
        Rv64ShiftWLogicalExecutor::new(Rv64BaseAluWU16AdapterExecutor, ShiftWOpcode::CLASS_OFFSET);
    let chip = Rv64ShiftWLogicalChip::<F>::new(
        ShiftWLogicalFiller::new(
            Rv64BaseAluWU16AdapterFiller::new(range_checker_chip.clone()),
            range_checker_chip,
            ShiftWOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_right_arithmetic_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64ShiftWRightArithmeticAir,
    Rv64ShiftWRightArithmeticExecutor,
    Rv64ShiftWRightArithmeticChip<F>,
) {
    let air = Rv64ShiftWRightArithmeticAir::new(
        Rv64BaseAluWU16AdapterAir::new(execution_bridge, memory_bridge, range_checker_chip.bus()),
        ShiftWRightArithmeticCoreAir::new(range_checker_chip.bus(), ShiftWOpcode::CLASS_OFFSET),
    );
    let executor = Rv64ShiftWRightArithmeticExecutor::new(
        Rv64BaseAluWU16AdapterExecutor::new(),
        ShiftWOpcode::CLASS_OFFSET,
    );
    let chip = Rv64ShiftWRightArithmeticChip::<F>::new(
        ShiftWRightArithmeticFiller::new(
            Rv64BaseAluWU16AdapterFiller::new(range_checker_chip.clone()),
            range_checker_chip,
            ShiftWOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_logical_harness(tester: &VmChipTestBuilder<F>) -> LogicalHarness {
    let (air, executor, chip) = create_logical_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
    );
    LogicalHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_right_arithmetic_harness(tester: &VmChipTestBuilder<F>) -> RightArithmeticHarness {
    let (air, executor, chip) = create_right_arithmetic_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
    );
    RightArithmeticHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftWOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    is_imm: Option<bool>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (c_imm, c) = if is_imm.unwrap_or(rng.random_bool(0.5)) {
        let (imm, c) = if let Some(c) = c {
            ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
        } else {
            generate_rv64_is_type_immediate(rng)
        };
        (Some(imm), c)
    } else {
        (
            None,
            c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX))),
        )
    };
    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        c_imm,
        opcode.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let b_word: [u8; RV64_WORD_NUM_LIMBS] = b[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let c_word: [u8; RV64_WORD_NUM_LIMBS] = c[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let (expected, _, _) = run_shift_w(opcode, &b_word, &c_word);
    assert_eq!(
        expected.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
    expected
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(SLLW, 100)]
#[test_case(SRLW, 100)]
#[test_case(SRAW, 100)]
fn run_rv64w_shift_rand_test(opcode: ShiftWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    if opcode == SRAW {
        let mut harness = create_right_arithmetic_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
                &mut rng,
                opcode,
                None,
                None,
                None,
            );
        }
        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    } else {
        let mut harness = create_logical_harness(&tester);
        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
                &mut rng,
                opcode,
                None,
                None,
                None,
            );
        }

        // Edge cases: shift by 0, by exactly one u16 limb, and across the limb boundary.
        for &shift in &[0u8, 1, 15, 16, 31] {
            let b = [0xAB, 0xCD, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
            let mut c = [0u8; RV64_REGISTER_NUM_LIMBS];
            c[0] = shift;
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.arena,
                &mut rng,
                opcode,
                Some(b),
                Some(false),
                Some(c),
            );
        }

        let tester = tester.build().load(harness).finalize();
        tester.simple_test().expect("Verification failed");
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

// ---- Logical (SLLW/SRLW) over the u16 core ----

#[derive(Clone, Copy, Default, PartialEq)]
struct LogicalShiftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: Option<[u32; NUM_LIMBS]>,
    pub bit_multiplier_left: Option<u32>,
    pub carry_multiplier_left: Option<u32>,
    /// Adapter sign bit of the low-word result.
    pub result_sign: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_aux: Option<[u32; NUM_LIMBS]>,
}

fn run_negative_shift_logical_test(
    opcode: ShiftWOpcode,
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: LogicalShiftPrankValues<RV64_WORD_U16_LIMBS, U16_BITS>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_logical_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64BaseAluWU16AdapterCols<F> = adapter_row.borrow_mut();
        let cols: &mut ShiftWLogicalCoreCols<F> = core_row.borrow_mut();

        if let Some(a) = prank_vals.a {
            cols.a = a.map(F::from_u32);
        }
        if let Some(bit_multiplier_left) = prank_vals.bit_multiplier_left {
            cols.bit_multiplier_left = F::from_u32(bit_multiplier_left);
        }
        if let Some(carry_multiplier_left) = prank_vals.carry_multiplier_left {
            cols.carry_multiplier_left = F::from_u32(carry_multiplier_left);
        }
        if let Some(result_sign) = prank_vals.result_sign {
            adapter_cols.result_sign = F::from_u32(result_sign);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_u32);
        }
        if let Some(bit_shift_aux) = prank_vals.bit_shift_aux {
            cols.bit_shift_aux = bit_shift_aux.map(F::from_u32);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_shiftw_logical_wrong_a_negative_test() {
    // b = 1, c = 1 (shift by 1). SLLW -> 2, SRLW -> 0; pranking a to 1 is wrong in both cases.
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        a: Some([1, 0]),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
    run_negative_shift_logical_test(SRLW, b, c, prank_vals);
}

#[test]
fn rv64_sllw_wrong_bit_carry_negative_test() {
    // low 32 bits all ones, shift by 9 bits. The high bits that cross the limb boundary are
    // nonzero; zeroing the carry breaks the decomposition (and the aux range check).
    let b = [255, 255, 255, 255, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        bit_shift_carry: Some([0; RV64_WORD_U16_LIMBS]),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
}

#[test]
fn rv64_sllw_wrong_bit_aux_negative_test() {
    // Zeroing the aux part breaks the b = aux + carry * 2^(16 - bit_shift) decomposition.
    let b = [255, 255, 255, 255, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        bit_shift_aux: Some([0; RV64_WORD_U16_LIMBS]),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
}

#[test]
fn rv64_sllw_wrong_limb_shift_negative_test() {
    let b = [1, 1, 0, 0, 0, 0, 0, 0];
    let c = [16, 0, 0, 0, 0, 0, 0, 0]; // shift by exactly one u16 limb
    let prank_vals = LogicalShiftPrankValues {
        limb_shift_marker: Some([1, 0]),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
}

#[test]
fn rv64_sllw_wrong_bit_mult_side_negative_test() {
    // For an SLLW row, force the multipliers onto the right-shift side: zeroing the SLL-gated
    // column makes the derived SRL-side multiplier become 2^9, and the output constraint fails.
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        bit_multiplier_left: Some(0),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
}

#[test]
fn rv64_srlw_wrong_bit_carry_negative_test() {
    let b = [255, 255, 255, 255, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        bit_shift_carry: Some([0; RV64_WORD_U16_LIMBS]),
        ..Default::default()
    };
    run_negative_shift_logical_test(SRLW, b, c, prank_vals);
}

#[test]
fn rv64_srlw_wrong_bit_mult_side_negative_test() {
    // For an SRLW row, setting the SLL-gated column to 2^9 zeroes the derived SRL-side
    // multiplier, so the multiplier-definition constraint fails.
    let b = [0, 0, 0, 128, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        bit_multiplier_left: Some(1 << 9),
        ..Default::default()
    };
    run_negative_shift_logical_test(SRLW, b, c, prank_vals);
}

// ---- Arithmetic right (SRAW) over the u16 core ----

#[derive(Clone, Copy, Default, PartialEq)]
struct ArithmeticShiftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: Option<[u32; NUM_LIMBS]>,
    pub b_sign: Option<u32>,
    /// Adapter sign bit of the low-word result.
    pub result_sign: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_aux: Option<[u32; NUM_LIMBS]>,
}

fn run_negative_shift_right_arithmetic_test(
    opcode: ShiftWOpcode,
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: ArithmeticShiftPrankValues<RV64_WORD_U16_LIMBS, U16_BITS>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_right_arithmetic_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64BaseAluWU16AdapterCols<F> = adapter_row.borrow_mut();
        let cols: &mut ShiftWRightArithmeticCoreCols<F> = core_row.borrow_mut();

        if let Some(a) = prank_vals.a {
            cols.a = a.map(F::from_u32);
        }
        if let Some(b_sign) = prank_vals.b_sign {
            cols.b_sign = F::from_u32(b_sign);
        }
        if let Some(result_sign) = prank_vals.result_sign {
            adapter_cols.result_sign = F::from_u32(result_sign);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_u32);
        }
        if let Some(bit_shift_aux) = prank_vals.bit_shift_aux {
            cols.bit_shift_aux = bit_shift_aux.map(F::from_u32);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_shiftw_wrong_negative_test() {
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    run_negative_shift_logical_test(
        SLLW,
        b,
        c,
        LogicalShiftPrankValues {
            a: Some([1, 0]),
            ..Default::default()
        },
    );
    run_negative_shift_logical_test(
        SRLW,
        b,
        c,
        LogicalShiftPrankValues {
            a: Some([1, 0]),
            ..Default::default()
        },
    );
    // SRAW(1, 1) = 0; pranking a to 1 is wrong.
    run_negative_shift_right_arithmetic_test(
        SRAW,
        b,
        c,
        ArithmeticShiftPrankValues {
            a: Some([1, 0]),
            ..Default::default()
        },
    );
}

#[test]
fn rv64_sraw_wrong_bit_shift_negative_test() {
    // b = 0x8000_0000 (negative word), shift by 9. Pranking bit_shift_marker to index 2 makes the
    // core encode a shift of 2, which disagrees with the register operand.
    let b = [0, 0, 0, 128, 255, 255, 255, 255];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let mut bit_shift_marker = [0u32; U16_BITS];
    bit_shift_marker[2] = 1;
    let prank_vals = ArithmeticShiftPrankValues {
        bit_shift_marker: Some(bit_shift_marker),
        ..Default::default()
    };
    run_negative_shift_right_arithmetic_test(SRAW, b, c, prank_vals);
}

#[test]
fn rv64_sraw_wrong_limb_shift_negative_test() {
    let b = [0, 0, 0, 128, 255, 255, 255, 255];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ArithmeticShiftPrankValues {
        limb_shift_marker: Some([0, 1]),
        ..Default::default()
    };
    run_negative_shift_right_arithmetic_test(SRAW, b, c, prank_vals);
}

#[test]
fn rv64_sraw_wrong_sign_negative_test() {
    // b is a negative word (top u16 limb sign bit set), so b_sign should be 1.
    let b = [0, 0, 0, 128, 255, 255, 255, 255];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ArithmeticShiftPrankValues {
        b_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_right_arithmetic_test(SRAW, b, c, prank_vals);
}

#[test]
fn rv64_shiftw_wrong_upper_sign_extension_negative_test() {
    // SLLW: b = 1 << 1 = 2, so the low-word result high limb has a zero sign bit; forcing
    // result_sign = 1 makes the adapter's sign-extension decomposition fail.
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = LogicalShiftPrankValues {
        result_sign: Some(1),
        ..Default::default()
    };
    run_negative_shift_logical_test(SLLW, b, c, prank_vals);
}

#[test]
fn rv64_shiftw_b_sign_only_prank_negative_test() {
    // SRAW with a zero shift: b_sign must still match the input sign bit.
    run_negative_shift_right_arithmetic_test(
        SRAW,
        [0, 0, 0, 128, 255, 255, 255, 255],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ArithmeticShiftPrankValues {
            b_sign: Some(0),
            ..Default::default()
        },
    );
}

#[test]
fn rv64_shiftw_result_sign_only_prank_negative_test() {
    // SLLW: result_sign must still match output sign bit/sign-extension.
    run_negative_shift_logical_test(
        SLLW,
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        LogicalShiftPrankValues {
            result_sign: Some(1),
            ..Default::default()
        },
    );

    // SRLW: result_sign must match output sign bit.
    run_negative_shift_logical_test(
        SRLW,
        [2, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        LogicalShiftPrankValues {
            result_sign: Some(1),
            ..Default::default()
        },
    );

    // SRAW: result_sign must match output sign bit. Negative word stays negative after a shift.
    run_negative_shift_right_arithmetic_test(
        SRAW,
        [0, 0, 0, 128, 255, 255, 255, 255],
        [1, 0, 0, 0, 0, 0, 0, 0],
        ArithmeticShiftPrankValues {
            result_sign: Some(0),
            ..Default::default()
        },
    );
}

#[test]
fn rv64_shiftw_wrong_upper_sign_extension_negative_to_zero_test() {
    let b = [0, 0, 0, 128, 255, 255, 255, 255];
    let c = [0, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ArithmeticShiftPrankValues {
        result_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_right_arithmetic_test(SRAW, b, c, prank_vals);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sllw_sanity_test() {
    // Inputs are sign-extended from 32-bit values. Result upper bytes sign-extend low 32-bit
    // result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [45, 7, 61, 186, 255, 255, 255, 255];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [91, 0, 100, 0, 0, 0, 0, 0];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [0, 0, 0, 104, 0, 0, 0, 0];
    let (result, limb_shift, bit_shift) = run_shift_w(
        SLLW,
        x[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
        y[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
    );
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

#[test]
fn run_srlw_sanity_test() {
    // Inputs are sign-extended from 32-bit values. Result upper bytes sign-extend low 32-bit
    // result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 255, 255, 255, 255];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [49, 190, 190, 190, 255, 255, 255, 255];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 100, 0, 0, 0, 0, 0, 0];
    let (result, limb_shift, bit_shift) = run_shift_w(
        SRLW,
        x[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
        y[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
    );
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

#[test]
fn run_sraw_sanity_test() {
    // Inputs are sign-extended from 32-bit values. Result upper bytes sign-extend low 32-bit
    // result.
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [31, 190, 221, 200, 255, 255, 255, 255];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [113, 20, 50, 80, 0, 0, 0, 0];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [110, 228, 255, 255, 255, 255, 255, 255];
    let (result, limb_shift, bit_shift) = run_shift_w(
        SRAW,
        x[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
        y[..RV64_WORD_NUM_LIMBS].try_into().unwrap(),
    );
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV64_WORD_NUM_LIMBS * RV64_BYTE_BITS);
    assert_eq!(shift / RV64_BYTE_BITS, limb_shift);
    assert_eq!(shift % RV64_BYTE_BITS, bit_shift);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuLogicalHarness = GpuTestChipHarness<
    F,
    Rv64ShiftWLogicalExecutor,
    Rv64ShiftWLogicalAir,
    Rv64ShiftWLogicalChipGpu,
    Rv64ShiftWLogicalChip<F>,
>;

#[cfg(feature = "cuda")]
type GpuRightArithmeticHarness = GpuTestChipHarness<
    F,
    Rv64ShiftWRightArithmeticExecutor,
    Rv64ShiftWRightArithmeticAir,
    Rv64ShiftWRightArithmeticChipGpu,
    Rv64ShiftWRightArithmeticChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_logical_harness(tester: &GpuChipTestBuilder) -> GpuLogicalHarness {
    let range_bus = default_var_range_checker_bus();
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_logical_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip =
        Rv64ShiftWLogicalChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_right_arithmetic_harness(tester: &GpuChipTestBuilder) -> GpuRightArithmeticHarness {
    let range_bus = default_var_range_checker_bus();
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_right_arithmetic_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip =
        Rv64ShiftWRightArithmeticChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(ShiftWOpcode::SLLW, 100)]
#[test_case(ShiftWOpcode::SRLW, 100)]
#[test_case(ShiftWOpcode::SRAW, 100)]
fn test_cuda_rand_shift_w_tracegen(opcode: ShiftWOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

    if opcode == SRAW {
        let mut harness = create_cuda_right_arithmetic_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                opcode,
                None,
                None,
                None,
            );
        }

        type Record<'a> = (
            &'a mut Rv64BaseAluWU16AdapterRecord,
            &'a mut ShiftRightArithmeticCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    } else {
        let mut harness = create_cuda_logical_harness(&tester);

        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut harness.executor,
                &mut harness.dense_arena,
                &mut rng,
                opcode,
                None,
                None,
                None,
            );
        }

        type Record<'a> = (
            &'a mut Rv64BaseAluWU16AdapterRecord,
            &'a mut ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv64BaseAluWU16AdapterExecutor>::new(),
            );

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
