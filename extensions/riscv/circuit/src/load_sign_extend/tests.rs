use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::VariableRangeCheckerChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS},
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
            default_bitwise_lookup_bus, default_var_range_checker_bus, dummy_range_checker,
            GpuChipTestBuilder, GpuTestChipHarness,
        },
        EmptyAdapterCoreLayout,
    },
};

use super::{core::LOAD_SIGN_EXTEND_CASES, run_write_data_sign_extend, LoadSignExtendCoreAir};
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

fn access_width(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADB => 1,
        LOADH => 2,
        LOADW => 4,
        _ => unreachable!(),
    }
}

/// One-hot flags array claiming the given `(opcode, shift)` case, with case index
/// `op_idx * 8 + shift` and op order [LOADB, LOADH, LOADW].
fn flags_for(opcode: Rv64LoadStoreOpcode, shift: usize) -> [bool; LOAD_SIGN_EXTEND_CASES] {
    let op_idx = match opcode {
        LOADB => 0,
        LOADH => 1,
        LOADW => 2,
        _ => unreachable!(),
    };
    array::from_fn(|i| i == op_idx * RV64_REGISTER_NUM_LIMBS + shift)
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    bitwise_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
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
        LoadSignExtendCoreAir::new(range_checker_chip.bus(), bitwise_chip.bus()),
    );
    let executor = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreAdapterExecutor::new(address_bits));
    let chip = Rv64LoadSignExtendChip::<F>::new(
        LoadSignExtendFiller::new(
            Rv64LoadStoreAdapterFiller::new(address_bits, range_checker_chip.clone()),
            range_checker_chip,
            bitwise_chip,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    (
        Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY),
        (bitwise_chip.air, bitwise_chip),
    )
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

    // Any byte shift is supported; keep the containing block and its successor in bounds.
    let ptr_val: u32 =
        rng.random_range(0..((1u32 << tester.address_bits()) - 2 * RV64_REGISTER_NUM_LIMBS as u32));
    // rs1 is 8 bytes, but only low 4 bytes used for address
    let rs1 = rs1.unwrap_or_else(|| {
        let low4 = ptr_val.wrapping_sub(imm_ext).to_le_bytes();
        [low4[0], low4[1], low4[2], low4[3], 0, 0, 0, 0]
    });
    let ptr_val = imm_ext.wrapping_add(rv64_bytes_to_u32(rs1));
    let a = gen_pointer(rng, 8);
    let b = gen_pointer(rng, 8);

    let shift_amount = (ptr_val % 8) as usize;
    let crosses = shift_amount + access_width(opcode) > RV64_REGISTER_NUM_LIMBS;
    tester.write_bytes(1, b, rs1.map(F::from_u8));

    let some_prev_data: [F; RV64_REGISTER_NUM_LIMBS] = if a != 0 {
        array::from_fn(|_| F::from_u8(rng.random()))
    } else {
        [F::ZERO; RV64_REGISTER_NUM_LIMBS]
    };
    let read_data: [u8; RV64_REGISTER_NUM_LIMBS] =
        read_data.unwrap_or(array::from_fn(|_| rng.random()));
    // Second block contents; only used (and only written to memory) when the access spans
    // two blocks.
    let block1_data: [u8; RV64_REGISTER_NUM_LIMBS] = array::from_fn(|_| rng.random());

    tester.write_bytes(1, a, some_prev_data);
    let aligned_ptr = (ptr_val as usize) - shift_amount;
    tester.write_bytes(2, aligned_ptr, read_data.map(F::from_u8));
    if crosses {
        tester.write_bytes(
            2,
            aligned_ptr + RV64_REGISTER_NUM_LIMBS,
            block1_data.map(F::from_u8),
        );
    }

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

    let read_data1 = if crosses {
        block1_data
    } else {
        [0; RV64_REGISTER_NUM_LIMBS]
    };
    let write_data = run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(
        opcode,
        read_data,
        read_data1,
        shift_amount,
    );
    if a != 0 {
        assert_eq!(write_data.map(F::from_u8), tester.read_bytes::<8>(1, a));
    } else {
        assert_eq!([F::ZERO; 8], tester.read_bytes::<8>(1, a));
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints. Shifts cover the full 0..8 range, including block-spanning
/// accesses.
///////////////////////////////////////////////////////////////////////////////////////
#[test_case(LOADB, 100)]
#[test_case(LOADH, 100)]
#[test_case(LOADW, 100)]
fn rand_load_sign_extend_test(opcode: Rv64LoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, bitwise) = create_test_chip(&mut tester);
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

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(LOADB, 7)]
#[test_case(LOADH, 3)]
#[test_case(LOADH, 7)]
#[test_case(LOADW, 2)]
#[test_case(LOADW, 5)]
#[test_case(LOADW, 7)]
fn positive_load_sign_extend_shift_test(opcode: Rv64LoadStoreOpcode, shift: u8) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_chip(&mut tester);

    // ptr = 64 + shift, imm = 0.
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        None,
        Some([64 + shift, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
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
    flags: Option<[bool; LOAD_SIGN_EXTEND_CASES]>,
    read_data1: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
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
    let (mut harness, bitwise) = create_test_chip(&mut tester);

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
        if let Some(read_data) = read_data {
            core_cols.read_data = read_data.map(F::from_u8);
        }
        if let Some(data_most_sig_bit) = prank_vals.data_most_sig_bit {
            core_cols.data_most_sig_bit = F::from_u32(data_most_sig_bit);
        }
        if let Some(flags) = prank_vals.flags {
            core_cols.flags = flags.map(F::from_bool);
        }
        if let Some(read_data1) = prank_vals.read_data1 {
            core_cols.read_data1 = read_data1.map(F::from_u8);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn loadstore_negative_tests() {
    // Pranking data_most_sig_bit breaks the sign-bit range check.
    run_negative_load_sign_extend_test(
        LOADB,
        Some([233, 187, 145, 238, 12, 55, 200, 99]),
        Some([64, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadSignExtPrankValues {
            data_most_sig_bit: Some(0),
            ..Default::default()
        },
        true,
    );

    // Claiming a different shift than the actual pointer breaks the adapter range check.
    run_negative_load_sign_extend_test(
        LOADH,
        None,
        Some([66, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadSignExtPrankValues {
            flags: Some(flags_for(LOADH, 0)),
            ..Default::default()
        },
        true,
    );

    // Claiming a different opcode than the program's breaks the execution bridge.
    run_negative_load_sign_extend_test(
        LOADH,
        None,
        Some([66, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadSignExtPrankValues {
            flags: Some(flags_for(LOADB, 2)),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn crossing_negative_tests() {
    // LOADW at ptr 64 + 5: spans two blocks. Denying the second block read (claiming the
    // non-crossing shift 1 instead) breaks the adapter range check.
    run_negative_load_sign_extend_test(
        LOADW,
        None,
        Some([69, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadSignExtPrankValues {
            flags: Some(flags_for(LOADW, 1)),
            ..Default::default()
        },
        true,
    );

    // Forging the second block's contents breaks the second block read interaction.
    run_negative_load_sign_extend_test(
        LOADW,
        None,
        Some([69, 0, 0, 0, 0, 0, 0, 0]),
        Some(0),
        Some(0),
        LoadSignExtPrankValues {
            read_data1: Some([1, 2, 3, 4, 5, 6, 7, 8]),
            ..Default::default()
        },
        true,
    );
}

fn solve(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u8; RV64_REGISTER_NUM_LIMBS],
    read_data1: [u8; RV64_REGISTER_NUM_LIMBS],
    shift: usize,
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    run_write_data_sign_extend::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(
        opcode, read_data, read_data1, shift,
    )
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn solve_loadh_extend_sign_sanity_test() {
    let read_data = [34, 159, 237, 151, 100, 200, 50, 25];
    let zero = [0; 8];
    assert_eq!(
        solve(LOADH, read_data, zero, 0),
        [34, 159, 255, 255, 255, 255, 255, 255]
    );
    assert_eq!(
        solve(LOADH, read_data, zero, 1),
        [159, 237, 255, 255, 255, 255, 255, 255]
    );
    assert_eq!(
        solve(LOADH, read_data, zero, 2),
        [237, 151, 255, 255, 255, 255, 255, 255]
    );
    assert_eq!(
        solve(LOADH, read_data, zero, 4),
        [100, 200, 255, 255, 255, 255, 255, 255]
    );
    assert_eq!(solve(LOADH, read_data, zero, 6), [50, 25, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn solve_loadh_extend_zero_sanity_test() {
    let read_data = [34, 121, 237, 97, 10, 20, 30, 40];
    let zero = [0; 8];
    assert_eq!(
        solve(LOADH, read_data, zero, 0),
        [34, 121, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        solve(LOADH, read_data, zero, 2),
        [237, 97, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(solve(LOADH, read_data, zero, 3), [97, 10, 0, 0, 0, 0, 0, 0]);
    assert_eq!(solve(LOADH, read_data, zero, 4), [10, 20, 0, 0, 0, 0, 0, 0]);
    assert_eq!(solve(LOADH, read_data, zero, 6), [30, 40, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn solve_loadb_sanity_test() {
    let read_data = [45, 82, 99, 127, 200, 150, 180, 210];
    for shift in 0..8 {
        let write_data = solve(LOADB, read_data, [0; 8], shift);
        let expected = (read_data[shift] as i8 as i64).to_le_bytes();
        assert_eq!(write_data, expected, "LOADB shift={shift}");
    }
}

#[test]
fn solve_loadw_sanity_test() {
    let read_data = [0x01, 0x02, 0x03, 0x84, 0xAA, 0xBB, 0xCC, 0x7D];
    let zero = [0; 8];
    // shift=0: word = 0x84030201 (negative)
    assert_eq!(
        solve(LOADW, read_data, zero, 0),
        [0x01, 0x02, 0x03, 0x84, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    // shift=2: word = [0x03, 0x84, 0xAA, 0xBB] => negative
    assert_eq!(
        solve(LOADW, read_data, zero, 2),
        [0x03, 0x84, 0xAA, 0xBB, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    // shift=4: word = [0xAA, 0xBB, 0xCC, 0x7D] => positive
    assert_eq!(
        solve(LOADW, read_data, zero, 4),
        [0xAA, 0xBB, 0xCC, 0x7D, 0x00, 0x00, 0x00, 0x00]
    );
}

#[test]
fn solve_crossing_sanity_test() {
    let read_data = [34, 159, 237, 151, 100, 200, 50, 25];
    let read_data1 = [200, 100, 3, 250, 66, 88, 120, 233];

    // LOADH at shift 7: bytes [25, 200] => negative halfword.
    assert_eq!(
        solve(LOADH, read_data, read_data1, 7),
        [25, 200, 255, 255, 255, 255, 255, 255]
    );
    // LOADB at shift 7 stays within the first block.
    assert_eq!(
        solve(LOADB, read_data, read_data1, 7),
        [25, 0, 0, 0, 0, 0, 0, 0]
    );
    // LOADW at shift 5: bytes [200, 50, 25, 200] => negative word.
    assert_eq!(
        solve(LOADW, read_data, read_data1, 5),
        [200, 50, 25, 200, 255, 255, 255, 255]
    );
    // LOADW at shift 6: bytes [50, 25, 200, 100] => positive word.
    assert_eq!(
        solve(LOADW, read_data, read_data1, 6),
        [50, 25, 200, 100, 0, 0, 0, 0]
    );
    // LOADW at shift 7: bytes [25, 200, 100, 3] => positive word.
    assert_eq!(
        solve(LOADW, read_data, read_data1, 7),
        [25, 200, 100, 3, 0, 0, 0, 0]
    );
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
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        default_bitwise_lookup_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv64LoadSignExtendChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
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
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

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
