use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftOpcode::{self, *};
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
        adapters::Rv64BaseAluRegU16AdapterRecord, Rv64ShiftLogicalChipGpu, ShiftLogicalCoreRecord,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    core::run_shift_logical, Rv64ShiftLogicalChip, ShiftLogicalCoreAir, ShiftLogicalCoreCols,
};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64BaseAluRegU16AdapterAir,
        Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller, RV64_REGISTER_NUM_LIMBS,
        U16_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
    Rv64ShiftLogicalAir, Rv64ShiftLogicalExecutor, ShiftLogicalFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const REGISTER_SHIFT_AMOUNTS: [u8; 8] = [0, 1, 15, 16, 31, 32, 63, 64];
type Harness =
    TestChipHarness<F, Rv64ShiftLogicalExecutor, Rv64ShiftLogicalAir, Rv64ShiftLogicalChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64ShiftLogicalAir,
    Rv64ShiftLogicalExecutor,
    Rv64ShiftLogicalChip<F>,
) {
    let air = Rv64ShiftLogicalAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        ShiftLogicalCoreAir::new(range_checker_chip.bus(), ShiftOpcode::CLASS_OFFSET),
    );
    let executor =
        Rv64ShiftLogicalExecutor::new(Rv64BaseAluRegU16AdapterExecutor, ShiftOpcode::CLASS_OFFSET);
    let chip = Rv64ShiftLogicalChip::<F>::new(
        ShiftLogicalFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let b_u16 = rv64_bytes_to_u16_block(b);
    let c_u16 = rv64_bytes_to_u16_block(c);
    let (a_u16, _, _) = run_shift_logical::<BLOCK_FE_WIDTH, U16_BITS>(opcode, &b_u16, &c_u16);
    let a_bytes = rv64_u16_block_to_bytes(a_u16);
    assert_eq!(
        a_bytes.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

fn execute_boundary_shifts<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
) {
    let b = [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x81];
    for shift in REGISTER_SHIFT_AMOUNTS {
        let mut c = [0u8; RV64_REGISTER_NUM_LIMBS];
        c[0] = shift;
        set_and_execute(tester, executor, arena, rng, opcode, Some(b), Some(c));
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////
#[test_case(SLL, 100)]
#[test_case(SRL, 100)]
fn run_rv64_shift_logical_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

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

    execute_boundary_shifts(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct ShiftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: Option<[u32; NUM_LIMBS]>,
    pub bit_multiplier_left: Option<u32>,
    pub carry_multiplier_left: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_aux: Option<[u32; NUM_LIMBS]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_shift_test(
    opcode: ShiftOpcode,
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_vals: ShiftPrankValues<BLOCK_FE_WIDTH, U16_BITS>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

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
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftLogicalCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(a) = prank_vals.a {
            cols.a = a.map(F::from_u32);
        }
        if let Some(bit_multiplier_left) = prank_vals.bit_multiplier_left {
            cols.bit_multiplier_left = F::from_u32(bit_multiplier_left);
        }
        if let Some(carry_multiplier_left) = prank_vals.carry_multiplier_left {
            cols.carry_multiplier_left = F::from_u32(carry_multiplier_left);
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
fn rv64_shift_logical_wrong_a_negative_test() {
    // b = 1, c = 1 (shift by 1). SLL -> 2, SRL -> 0; pranking a to 1 is wrong in both cases.
    let b = [1, 0, 0, 0, 0, 0, 0, 0];
    let c = [1, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        a: Some([1, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, b, c, prank_vals, false);
    run_negative_shift_test(SRL, b, c, prank_vals, false);
}

#[test]
fn rv64_sll_wrong_bit_carry_negative_test() {
    // b = all 0xFFFF, shift by 9 bits. The high bits that cross the limb boundary are nonzero;
    // zeroing the carry breaks the decomposition (and the aux range check).
    let b = [255; RV64_REGISTER_NUM_LIMBS];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_carry: Some([0; BLOCK_FE_WIDTH]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, b, c, prank_vals, true);
}

#[test]
fn rv64_sll_wrong_bit_aux_negative_test() {
    // Zeroing the aux part breaks the b = aux + carry * 2^(16 - bit_shift) decomposition.
    let b = [255; RV64_REGISTER_NUM_LIMBS];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_aux: Some([0; BLOCK_FE_WIDTH]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, b, c, prank_vals, false);
}

#[test]
fn rv64_sll_wrong_limb_shift_negative_test() {
    let b = [1, 1, 0, 0, 0, 0, 0, 0];
    let c = [16, 0, 0, 0, 0, 0, 0, 0]; // shift by exactly one u16 limb
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 0, 1, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, b, c, prank_vals, false);
}

#[test]
fn rv64_sll_wrong_bit_multiplier_negative_test() {
    // For an SLL row, force the multipliers onto the right-shift side: zeroing the SLL-gated
    // column makes the derived SRL-side multiplier become 2^9, and the output constraint fails.
    let b = [1, 1, 1, 1, 0, 0, 0, 0];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SLL, b, c, prank_vals, false);
}

#[test]
fn rv64_srl_wrong_bit_carry_negative_test() {
    let b = [255; RV64_REGISTER_NUM_LIMBS];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_carry: Some([0; BLOCK_FE_WIDTH]),
        ..Default::default()
    };
    run_negative_shift_test(SRL, b, c, prank_vals, true);
}

#[test]
fn rv64_srl_wrong_bit_multiplier_negative_test() {
    // For an SRL row, setting the SLL-gated column to 2^9 zeroes the derived SRL-side
    // multiplier, so the multiplier-definition constraint fails.
    let b = [0, 0, 0, 0, 0, 0, 0, 128];
    let c = [9, 0, 0, 0, 0, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(1 << 9),
        ..Default::default()
    };
    run_negative_shift_test(SRL, b, c, prank_vals, false);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sll_sanity_test() {
    let x = rv64_bytes_to_u16_block([45, 7, 61, 186, 31, 190, 221, 200]);
    let y = rv64_bytes_to_u16_block([91, 0, 100, 0, 49, 190, 190, 113]);
    let (result, limb_shift, bit_shift) =
        run_shift_logical::<BLOCK_FE_WIDTH, U16_BITS>(SLL, &x, &y);
    // Reference: shift the full 64-bit value left by (y[0] % 64) bits.
    let expected =
        (u64::from_le_bytes([45, 7, 61, 186, 31, 190, 221, 200]) << (91u32 % 64)).to_le_bytes();
    assert_eq!(rv64_u16_block_to_bytes(result), expected);
    let shift = (y[0] as usize) % (BLOCK_FE_WIDTH * U16_BITS);
    assert_eq!(shift / U16_BITS, limb_shift);
    assert_eq!(shift % U16_BITS, bit_shift);
}

#[test]
fn run_srl_sanity_test() {
    let x = rv64_bytes_to_u16_block([31, 190, 221, 200, 45, 7, 61, 186]);
    let y = rv64_bytes_to_u16_block([81, 190, 190, 190, 113, 20, 50, 80]);
    let (result, limb_shift, bit_shift) =
        run_shift_logical::<BLOCK_FE_WIDTH, U16_BITS>(SRL, &x, &y);
    let expected =
        (u64::from_le_bytes([31, 190, 221, 200, 45, 7, 61, 186]) >> (81u32 % 64)).to_le_bytes();
    assert_eq!(rv64_u16_block_to_bytes(result), expected);
    let shift = (y[0] as usize) % (BLOCK_FE_WIDTH * U16_BITS);
    assert_eq!(shift / U16_BITS, limb_shift);
    assert_eq!(shift % U16_BITS, bit_shift);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64ShiftLogicalExecutor,
    Rv64ShiftLogicalAir,
    Rv64ShiftLogicalChipGpu,
    Rv64ShiftLogicalChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64ShiftLogicalChipGpu::new(
        tester.range_checker(),
        tester.timestamp_max_bits(),
        Default::default(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(ShiftOpcode::SLL, 100)]
#[test_case(ShiftOpcode::SRL, 100)]
fn test_cuda_rand_shift_logical_tracegen(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

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

    execute_boundary_shifts(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        opcode,
    );

    type Record<'a> = (
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut ShiftLogicalCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
