use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
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
        adapters::Rv64LoadStoreAdapterRecord, LoadSignExtendCoreRecord, Rv64LoadSignExtendChipGpu,
    },
    openvm_circuit::arch::{
        testing::{
            default_var_range_checker_bus, dummy_range_checker, GpuChipTestBuilder,
            GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::{run_write_data_sign_extend, LoadSignExtendCoreAir};
use crate::{
    adapters::{
        rv64_bytes_to_u32, Rv64LoadStoreAdapterAir, Rv64LoadStoreAdapterExecutor,
        Rv64LoadStoreAdapterFiller,
    },
    load_sign_extend::LoadSignExtendCoreCols,
    LoadSignExtendFiller, Rv64LoadSignExtendAir, Rv64LoadSignExtendChip,
    Rv64LoadSignExtendExecutor,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<
    F,
    Rv64LoadSignExtendExecutor,
    Rv64LoadSignExtendAir,
    Rv64LoadSignExtendChip<F>,
>;
type F = BabyBear;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64LoadSignExtendAir,
    Rv64LoadSignExtendExecutor,
    Rv64LoadSignExtendChip<F>,
) {
    let air = Rv64LoadSignExtendAir::new(
        Rv64LoadStoreAdapterAir::new(
            memory_bridge,
            execution_bridge,
            range_checker_chip.bus(),
            address_bits,
        ),
        LoadSignExtendCoreAir::new(range_checker_chip.bus()),
    );
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreAdapterExecutor::new(address_bits));
    let chip = Rv64LoadSignExtendChip::<F>::new(
        LoadSignExtendFiller::new(
            Rv64LoadStoreAdapterFiller::new(address_bits, range_checker_chip.clone()),
            range_checker_chip,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(tester: &mut VmChipTestBuilder<F>) -> Harness {
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64LoadStoreOpcode,
    read_data: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    rs1: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.random_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or(rng.random_range(0..2));
    let imm_ext = imm + imm_sign * (0xffff0000);

    let alignment = match opcode {
        LOADB => 0,
        LOADH => 1,
        LOADW => 2,
        _ => unreachable!(),
    };

    let ptr_val: u32 = rng.random_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    // rs1 is 8 bytes, but only low 4 bytes used for address
    let rs1 = rs1.unwrap_or_else(|| {
        let low4 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
        [low4[0], low4[1], low4[2], low4[3], 0, 0, 0, 0]
    });
    let ptr_val = imm_ext.wrapping_add(rv64_bytes_to_u32(rs1));
    let a = gen_pointer(rng, 8);
    let b = gen_pointer(rng, 8);

    let shift_amount = ptr_val % 8;
    tester.write(1, b, rs1.map(F::from_u8));

    let some_prev_data: [F; RV64_REGISTER_NUM_LIMBS] = if a != 0 {
        array::from_fn(|_| F::from_u8(rng.random()))
    } else {
        [F::ZERO; RV64_REGISTER_NUM_LIMBS]
    };
    let read_data: [u8; RV64_REGISTER_NUM_LIMBS] =
        read_data.unwrap_or(array::from_fn(|_| rng.random()));

    tester.write(1, a, some_prev_data);
    tester.write(
        2,
        (ptr_val - shift_amount) as usize,
        read_data.map(F::from_u8),
    );

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b,
                imm as usize,
                1,
                2,
                (a != 0) as usize,
                imm_sign as usize,
            ],
        ),
    );

    let write_data = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
        opcode,
        read_data,
        shift_amount as usize,
    );
    if a != 0 {
        assert_eq!(write_data.map(F::from_u8), tester.read::<8>(1, a));
    } else {
        assert_eq!([F::ZERO; 8], tester.read::<8>(1, a));
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test_case(LOADB, 100)]
#[test_case(LOADH, 100)]
#[test_case(LOADW, 100)]
fn rand_load_sign_extend_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_test_chip(&mut tester);
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
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_loadb_shift7_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADB,
        None,
        Some([7, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_loadh_shift6_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADH,
        None,
        Some([6, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn positive_loadw_shift4_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        LOADW,
        None,
        Some([4, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
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
struct LoadSignExtPrankValues {
    data_most_sig_bit: Option<u32>,
    shift_most_sig_bit: Option<u32>,
    opcode_flags: Option<[bool; 7]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_load_sign_extend_test(
    opcode: Rv64LoadStoreOpcode,
    read_data: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    rs1: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    prank_vals: LoadSignExtPrankValues,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        read_data,
        rs1,
        imm,
        imm_sign,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut LoadSignExtendCoreCols<F, RV64_REGISTER_NUM_LIMBS> =
            core_row.borrow_mut();
        if let Some(shifted_read_data) = read_data {
            core_cols.shifted_read_data = shifted_read_data.map(F::from_u8);
        }
        if let Some(data_most_sig_bit) = prank_vals.data_most_sig_bit {
            core_cols.data_most_sig_bit = F::from_u32(data_most_sig_bit);
        }
        if let Some(shift_most_sig_bit) = prank_vals.shift_most_sig_bit {
            core_cols.shift_most_sig_bit = F::from_u32(shift_most_sig_bit);
        }
        if let Some(opcode_flags) = prank_vals.opcode_flags {
            core_cols.opcode_loadb_flag0 = F::from_bool(opcode_flags[0]);
            core_cols.opcode_loadb_flag1 = F::from_bool(opcode_flags[1]);
            core_cols.opcode_loadb_flag2 = F::from_bool(opcode_flags[2]);
            core_cols.opcode_loadb_flag3 = F::from_bool(opcode_flags[3]);
            core_cols.opcode_loadh_flag0 = F::from_bool(opcode_flags[4]);
            core_cols.opcode_loadh_flag2 = F::from_bool(opcode_flags[5]);
            core_cols.opcode_loadw_flag = F::from_bool(opcode_flags[6]);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
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
fn loadstore_negative_tests() {
    // Pranking shifted_read_data + data_most_sig_bit breaks memory read interaction
    run_negative_load_sign_extend_test(
        LOADB,
        Some([233, 187, 145, 238, 12, 55, 200, 99]),
        None,
        None,
        None,
        LoadSignExtPrankValues {
            data_most_sig_bit: Some(0),
            ..Default::default()
        },
        true,
    );

    // Pranking shift_most_sig_bit breaks the read_data unrotation → memory interaction error
    run_negative_load_sign_extend_test(
        LOADH,
        None,
        Some([202, 109, 183, 26, 0, 0, 0, 0]),
        Some(31212),
        None,
        LoadSignExtPrankValues {
            shift_most_sig_bit: Some(0),
            ..Default::default()
        },
        true,
    );

    // Pranking opcode_flags to wrong value → execution bridge interaction error
    run_negative_load_sign_extend_test(
        LOADB,
        None,
        Some([250, 132, 77, 5, 0, 0, 0, 0]),
        Some(47741),
        None,
        LoadSignExtPrankValues {
            opcode_flags: Some([true, false, false, false, false, false, false]),
            ..Default::default()
        },
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn solve_loadh_extend_sign_sanity_test() {
    let read_data = [34, 159, 237, 151, 100, 200, 50, 25];
    let write_data0 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 0);
    let write_data2 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 2);
    let write_data4 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 4);
    let write_data6 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 6);

    assert_eq!(write_data0, [34, 159, 255, 255, 255, 255, 255, 255]);
    assert_eq!(write_data2, [237, 151, 255, 255, 255, 255, 255, 255]);
    assert_eq!(write_data4, [100, 200, 255, 255, 255, 255, 255, 255]);
    assert_eq!(write_data6, [50, 25, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn solve_loadh_extend_zero_sanity_test() {
    let read_data = [34, 121, 237, 97, 10, 20, 30, 40];
    let write_data0 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 0);
    let write_data2 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 2);
    let write_data4 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 4);
    let write_data6 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 6);

    assert_eq!(write_data0, [34, 121, 0, 0, 0, 0, 0, 0]);
    assert_eq!(write_data2, [237, 97, 0, 0, 0, 0, 0, 0]);
    assert_eq!(write_data4, [10, 20, 0, 0, 0, 0, 0, 0]);
    assert_eq!(write_data6, [30, 40, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn solve_loadb_extend_sign_sanity_test() {
    let read_data = [45, 82, 99, 127, 200, 150, 180, 210];
    for shift in 0..8 {
        let write_data = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
            LOADB, read_data, shift,
        );
        let byte = read_data[shift];
        let expected = (byte as i8 as i64).to_le_bytes();
        assert_eq!(write_data, expected, "LOADB shift={shift}");
    }
}

#[test]
fn solve_loadb_extend_zero_sanity_test() {
    let read_data = [173, 210, 227, 255, 128, 250, 200, 190];
    for shift in 0..8 {
        let write_data = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
            LOADB, read_data, shift,
        );
        let byte = read_data[shift];
        let expected = (byte as i8 as i64).to_le_bytes();
        assert_eq!(write_data, expected, "LOADB shift={shift}");
    }
}

#[test]
fn solve_loadw_extend_sign_sanity_test() {
    // shift=0: word = [0x01, 0x02, 0x03, 0x84] => 0x84030201 (negative)
    let read_data = [0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0xDD];
    let write_data0 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data, 0);
    assert_eq!(
        write_data0,
        [0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF]
    );

    // shift=4: word = [0xAA, 0xBB, 0xCC, 0xDD] => 0xDDCCBBAA (negative)
    let write_data4 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data, 4);
    assert_eq!(
        write_data4,
        [0xAA, 0xBB, 0xCC, 0xDD, 0xFF, 0xFF, 0xFF, 0xFF]
    );
}

#[test]
fn solve_loadw_extend_zero_sanity_test() {
    // shift=0: word = [0x01, 0x02, 0x03, 0x04] => 0x04030201 (positive)
    let read_data = [0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
    let write_data0 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data, 0);
    assert_eq!(
        write_data0,
        [0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]
    );

    // shift=4: word = [0xAA, 0xBB, 0xCC, 0x7D] => 0x7DCCBBAA (positive)
    let read_data2 = [0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0x7D];
    let write_data4 =
        run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data2, 4);
    assert_eq!(
        write_data4,
        [0xAA, 0xBB, 0xCC, 0x7D, 0x00, 0x00, 0x00, 0x00]
    );
}

#[test]
#[should_panic(expected = "LOADW requires 4-byte aligned shift")]
fn solve_loadw_rejects_shift_2() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data, 2);
}

#[test]
#[should_panic(expected = "LOADW requires 4-byte aligned shift")]
fn solve_loadw_rejects_shift_6() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADW, read_data, 6);
}

#[test]
#[should_panic(expected = "LOADH requires 2-byte aligned shift")]
fn solve_loadh_rejects_shift_1() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 1);
}

#[test]
#[should_panic(expected = "LOADH requires 2-byte aligned shift")]
fn solve_loadh_rejects_shift_3() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 3);
}

#[test]
#[should_panic(expected = "LOADH requires 2-byte aligned shift")]
fn solve_loadh_rejects_shift_5() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 5);
}

#[test]
#[should_panic(expected = "LOADH requires 2-byte aligned shift")]
fn solve_loadh_rejects_shift_7() {
    let read_data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(LOADH, read_data, 7);
}

/// Assert the full set of accepted shifts per opcode:
/// LOADB: 0..7, LOADH: {0,2,4,6}, LOADW: {0,4}
#[test]
fn accepted_shift_sets() {
    let read_data = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];

    // LOADB accepts all shifts 0..7
    for shift in 0..8 {
        let _ = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
            LOADB, read_data, shift,
        );
    }

    // LOADH accepts even shifts {0, 2, 4, 6}
    for shift in [0, 2, 4, 6] {
        let _ = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
            LOADH, read_data, shift,
        );
    }

    // LOADW accepts only {0, 4}
    for shift in [0, 4] {
        let _ = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(
            LOADW, read_data, shift,
        );
    }
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64LoadSignExtendExecutor,
    Rv64LoadSignExtendAir,
    Rv64LoadSignExtendChipGpu,
    Rv64LoadSignExtendChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = dummy_range_checker(default_var_range_checker_bus());

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv64LoadSignExtendChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADB, 100)]
#[test_case(LOADH, 100)]
#[test_case(LOADW, 100)]
fn test_cuda_rand_load_sign_extend_tracegen(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
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
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64LoadStoreAdapterRecord,
        &'a mut LoadSignExtendCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64LoadStoreAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
