use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::FieldAlgebra,
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
        adapters::Rv32LoadStoreAdapterRecord, LoadSignExtendCoreRecord, Rv32LoadSignExtendChipGpu,
    },
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{run_write_data_sign_extend, LoadSignExtendCoreAir};
use crate::{
    adapters::{
        Rv32LoadStoreAdapterAir, Rv32LoadStoreAdapterExecutor, Rv32LoadStoreAdapterFiller,
        RV32_REGISTER_NUM_LIMBS,
    },
    load_sign_extend::LoadSignExtendCoreCols,
    test_utils::get_verification_error,
    LoadSignExtendFiller, Rv32LoadSignExtendAir, Rv32LoadSignExtendChip,
    Rv32LoadSignExtendExecutor,
};

const IMM_BITS: usize = 16;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<
    F,
    Rv32LoadSignExtendExecutor,
    Rv32LoadSignExtendAir,
    Rv32LoadSignExtendChip<F>,
>;
type F = BabyBear;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: Arc<VariableRangeCheckerChip>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv32LoadSignExtendAir,
    Rv32LoadSignExtendExecutor,
    Rv32LoadSignExtendChip<F>,
) {
    let air = Rv32LoadSignExtendAir::new(
        Rv32LoadStoreAdapterAir::new(
            memory_bridge,
            execution_bridge,
            range_checker_chip.bus(),
            address_bits,
        ),
        LoadSignExtendCoreAir::new(range_checker_chip.bus()),
    );
    let executor = Rv32LoadSignExtendExecutor::new(Rv32LoadStoreAdapterExecutor::new(address_bits));
    let chip = Rv32LoadSignExtendChip::<F>::new(
        LoadSignExtendFiller::new(
            Rv32LoadStoreAdapterFiller::new(address_bits, range_checker_chip.clone()),
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
    opcode: Rv32LoadStoreOpcode,
    read_data: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    rs1: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_sign = imm_sign.unwrap_or(rng.gen_range(0..2));
    let imm_ext = imm + imm_sign * (0xffff0000);

    let alignment = match opcode {
        LOADB => 0,
        LOADH => 1,
        _ => unreachable!(),
    };

    let ptr_val: u32 = rng.gen_range(0..(1 << (tester.address_bits() - alignment))) << alignment;
    let rs1 = rs1.unwrap_or(ptr_val.wrapping_sub(imm_ext).to_le_bytes());
    let ptr_val = imm_ext.wrapping_add(u32::from_le_bytes(rs1));
    let a = gen_pointer(rng, 4);
    let b = gen_pointer(rng, 4);

    let shift_amount = ptr_val % 4;
    tester.write(1, b, rs1.map(F::from_canonical_u8));

    let some_prev_data: [F; RV32_REGISTER_NUM_LIMBS] = if a != 0 {
        array::from_fn(|_| F::from_canonical_u8(rng.gen()))
    } else {
        [F::ZERO; RV32_REGISTER_NUM_LIMBS]
    };
    let read_data: [u8; RV32_REGISTER_NUM_LIMBS] =
        read_data.unwrap_or(array::from_fn(|_| rng.gen()));

    tester.write(1, a, some_prev_data);
    tester.write(
        2,
        (ptr_val - shift_amount) as usize,
        read_data.map(F::from_canonical_u8),
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

    let write_data = run_write_data_sign_extend(opcode, read_data, shift_amount as usize);
    if a != 0 {
        assert_eq!(write_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
    } else {
        assert_eq!([F::ZERO; 4], tester.read::<4>(1, a));
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
fn rand_load_sign_extend_test(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
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
    opcode_flags: Option<[bool; 3]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_load_sign_extend_test(
    opcode: Rv32LoadStoreOpcode,
    read_data: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    rs1: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    imm_sign: Option<u32>,
    prank_vals: LoadSignExtPrankValues,
    interaction_error: bool,
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
        let mut trace_row = trace.row_slice(0).to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut LoadSignExtendCoreCols<F, RV32_REGISTER_NUM_LIMBS> =
            core_row.borrow_mut();
        if let Some(shifted_read_data) = read_data {
            core_cols.shifted_read_data = shifted_read_data.map(F::from_canonical_u8);
        }
        if let Some(data_most_sig_bit) = prank_vals.data_most_sig_bit {
            core_cols.data_most_sig_bit = F::from_canonical_u32(data_most_sig_bit);
        }
        if let Some(shift_most_sig_bit) = prank_vals.shift_most_sig_bit {
            core_cols.shift_most_sig_bit = F::from_canonical_u32(shift_most_sig_bit);
        }
        if let Some(opcode_flags) = prank_vals.opcode_flags {
            core_cols.opcode_loadb_flag0 = F::from_bool(opcode_flags[0]);
            core_cols.opcode_loadb_flag1 = F::from_bool(opcode_flags[1]);
            core_cols.opcode_loadh_flag = F::from_bool(opcode_flags[2]);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn loadstore_negative_tests() {
    run_negative_load_sign_extend_test(
        LOADB,
        Some([233, 187, 145, 238]),
        None,
        None,
        None,
        LoadSignExtPrankValues {
            data_most_sig_bit: Some(0),
            ..Default::default()
        },
        true,
    );

    run_negative_load_sign_extend_test(
        LOADH,
        None,
        Some([202, 109, 183, 26]),
        Some(31212),
        None,
        LoadSignExtPrankValues {
            shift_most_sig_bit: Some(0),
            ..Default::default()
        },
        true,
    );

    run_negative_load_sign_extend_test(
        LOADB,
        None,
        Some([250, 132, 77, 5]),
        Some(47741),
        None,
        LoadSignExtPrankValues {
            opcode_flags: Some([true, false, false]),
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
    let read_data = [34, 159, 237, 151];
    let write_data0 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADH, read_data, 0);
    let write_data2 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADH, read_data, 2);

    assert_eq!(write_data0, [34, 159, 255, 255]);
    assert_eq!(write_data2, [237, 151, 255, 255]);
}

#[test]
fn solve_loadh_extend_zero_sanity_test() {
    let read_data = [34, 121, 237, 97];
    let write_data0 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADH, read_data, 0);
    let write_data2 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADH, read_data, 2);

    assert_eq!(write_data0, [34, 121, 0, 0]);
    assert_eq!(write_data2, [237, 97, 0, 0]);
}

#[test]
fn solve_loadb_extend_sign_sanity_test() {
    let read_data = [45, 82, 99, 127];
    let write_data0 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 0);
    let write_data1 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 1);
    let write_data2 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 2);
    let write_data3 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 3);

    assert_eq!(write_data0, [45, 0, 0, 0]);
    assert_eq!(write_data1, [82, 0, 0, 0]);
    assert_eq!(write_data2, [99, 0, 0, 0]);
    assert_eq!(write_data3, [127, 0, 0, 0]);
}

#[test]
fn solve_loadb_extend_zero_sanity_test() {
    let read_data = [173, 210, 227, 255];
    let write_data0 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 0);
    let write_data1 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 1);
    let write_data2 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 2);
    let write_data3 = run_write_data_sign_extend::<RV32_REGISTER_NUM_LIMBS>(LOADB, read_data, 3);

    assert_eq!(write_data0, [173, 255, 255, 255]);
    assert_eq!(write_data1, [210, 255, 255, 255]);
    assert_eq!(write_data2, [227, 255, 255, 255]);
    assert_eq!(write_data3, [255, 255, 255, 255]);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv32LoadSignExtendExecutor,
    Rv32LoadSignExtendAir,
    Rv32LoadSignExtendChipGpu,
    Rv32LoadSignExtendChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_bus = default_var_range_checker_bus();
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv32LoadSignExtendChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LOADB, 100)]
#[test_case(LOADH, 100)]
fn test_cuda_rand_load_sign_extend_tracegen(opcode: Rv32LoadStoreOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
    let mut rng = create_seeded_rng();

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
        &'a mut Rv32LoadStoreAdapterRecord,
        &'a mut LoadSignExtendCoreRecord<RV32_REGISTER_NUM_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32LoadStoreAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
