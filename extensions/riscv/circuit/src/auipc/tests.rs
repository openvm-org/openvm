use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor, VmAirWrapper, VmChipWrapper,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_riscv_transpiler::Rv64AuipcOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits,
                RvrPreflightTranscript,
            },
            testing::default_var_range_checker_bus,
            MatrixRecordArena, VmExecutor,
        },
        system::{
            cuda::memory::MemoryInventoryGPU,
            memory::online::{AddressMap, GuestMemory, TracingMemory},
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::Chip,
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        base::DeviceMatrix,
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        stream::device_synchronize,
    },
    openvm_instructions::{
        exe::VmExe,
        program::Program,
        riscv::{RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
        SystemOpcode,
    },
    openvm_stark_backend::prover::ColMajorMatrix,
};
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64RdWriteAdapterRecord, Rv64AuipcChipGpu, Rv64AuipcCoreRecord},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use crate::{
    adapters::{
        rv64_u16_block_to_bytes, Rv64RdWriteAdapterAir, Rv64RdWriteAdapterCols,
        Rv64RdWriteAdapterExecutor, Rv64RdWriteAdapterFiller, RV64_BYTE_BITS, RV64_PTR_U16_LIMBS,
        RV64_WORD_NUM_LIMBS,
    },
    auipc::{run_auipc, Rv64AuipcCoreCols},
    Rv64AuipcAir, Rv64AuipcChip, Rv64AuipcCoreAir, Rv64AuipcExecutor, Rv64AuipcFiller,
};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AuipcExecutor, Rv64AuipcAir, Rv64AuipcChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AuipcAir, Rv64AuipcExecutor, Rv64AuipcChip<F>) {
    let air = VmAirWrapper::new(
        Rv64RdWriteAdapterAir::new(memory_bridge, execution_bridge),
        Rv64AuipcCoreAir::new(range_checker_chip.bus()),
    );
    let executor = Rv64AuipcExecutor::new(Rv64RdWriteAdapterExecutor::new());
    let chip = VmChipWrapper::<F, _>::new(
        Rv64AuipcFiller::new(Rv64RdWriteAdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
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
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64AuipcOpcode,
    imm: Option<u32>,
    initial_pc: Option<u32>,
) where
    Rv64AuipcExecutor: PreflightExecutor<F, RA>,
{
    let imm = imm.unwrap_or(rng.random_range(0..(1 << IMM_BITS))) as usize;
    let a = rng.random_range(0..32) << 3;

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [a, 0, imm, 1, 0]),
        initial_pc.unwrap_or(rng.random_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.last_from_pc().as_canonical_u32();
    let rd_data = run_auipc(initial_pc, imm as u32);
    let rd_bytes = rv64_u16_block_to_bytes(rd_data);
    assert_eq!(rd_bytes.map(F::from_u8), tester.read_bytes::<8>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_auipc_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            AUIPC,
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

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct AuipcPrankValues {
    pub is_sign_extend: Option<u32>,
    pub rd_data: Option<[u32; RV64_PTR_U16_LIMBS]>,
    pub imm_low_8: Option<u32>,
    pub imm_high_16: Option<u32>,
    pub pc_high: Option<u32>,
}

fn pack_rd_u8_limbs(limbs: [u32; RV64_WORD_NUM_LIMBS]) -> [u32; RV64_PTR_U16_LIMBS] {
    [
        limbs[0] + (limbs[1] << RV64_BYTE_BITS),
        limbs[2] + (limbs[3] << RV64_BYTE_BITS),
    ]
}

fn split_imm_u8_limbs(limbs: [u32; RV64_WORD_NUM_LIMBS - 1]) -> (u32, u32) {
    (limbs[0], limbs[1] + (limbs[2] << RV64_BYTE_BITS))
}

fn run_negative_auipc_test(
    opcode: Rv64AuipcOpcode,
    initial_imm: Option<u32>,
    initial_pc: Option<u32>,
    prank_vals: AuipcPrankValues,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        initial_imm,
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core_cols: &mut Rv64AuipcCoreCols<F> = core_row.borrow_mut();

        if let Some(val) = prank_vals.is_sign_extend {
            core_cols.is_sign_extend = F::from_u32(val);
        }
        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_u32);
        }
        if let Some(val) = prank_vals.imm_low_8 {
            core_cols.imm_low_8 = F::from_u32(val);
        }
        if let Some(val) = prank_vals.imm_high_16 {
            core_cols.imm_high_16 = F::from_u32(val);
        }
        if let Some(val) = prank_vals.pc_high {
            core_cols.pc_high = F::from_u32(val);
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
fn invalid_limb_negative_tests() {
    let (imm_low_8, imm_high_16) = split_imm_u8_limbs([107, 46, 81]);
    run_negative_auipc_test(
        AUIPC,
        Some(9722891),
        None,
        AuipcPrankValues {
            imm_low_8: Some(imm_low_8),
            imm_high_16: Some(imm_high_16),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(2110400),
        AuipcPrankValues {
            rd_data: Some(pack_rd_u8_limbs([194, 51, 32, 240])),
            ..Default::default()
        },
        true,
    );
    let (_, imm_high_16) = split_imm_u8_limbs([0, 206, 166]);
    run_negative_auipc_test(
        AUIPC,
        None,
        None,
        AuipcPrankValues {
            imm_high_16: Some(imm_high_16),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        None,
        None,
        AuipcPrankValues {
            rd_data: Some(pack_rd_u8_limbs([30, 92, 82, 132])),
            ..Default::default()
        },
        false,
    );
    let (imm_low_8, imm_high_16) = split_imm_u8_limbs([166, 243, 17]);
    run_negative_auipc_test(
        AUIPC,
        None,
        Some(876487877),
        AuipcPrankValues {
            rd_data: Some(pack_rd_u8_limbs([197, 202, 49, 70])),
            imm_low_8: Some(imm_low_8),
            imm_high_16: Some(imm_high_16),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn rd_upper_bytes_trace_tamper_negative_test() {
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    let initial_pc = 0x1234;
    let imm = 16usize;
    let rd_ptr = 16usize;

    let clean_rd_prev = [9u32, 8, 7, 6, 0, 0, 0, 0];

    // Seed the destination register with a known clean value.
    tester.write_bytes(1, rd_ptr, clean_rd_prev.map(F::from_u32));

    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(AUIPC.global_opcode(), [rd_ptr, 0, imm, 1, 0]),
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let (adapter_row, _) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv64RdWriteAdapterCols<F> = adapter_row.borrow_mut();
        adapter_cols.rd_aux_cols.prev_data[1] = F::from_u32(1);
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
fn sign_extend_flag_negative_tests() {
    // Prank is_sign_extend = 1 when the result has bit 31 unset (MSB of rd_data[1] is 0).
    // pc=4, imm=0 ⟹ rd = 4 ⟹ rd low 32 bits = [4, 0].
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(4),
        AuipcPrankValues {
            is_sign_extend: Some(1),
            ..Default::default()
        },
        true,
    );
    // Prank is_sign_extend = 0 when the result has bit 31 set (MSB of rd_data[1] is 1).
    // pc=0, imm=2^23 ⟹ rd = 2^31 ⟹ rd low 32 bits = [0, 0x8000].
    run_negative_auipc_test(
        AUIPC,
        Some(1 << 23),
        Some(0),
        AuipcPrankValues {
            is_sign_extend: Some(0),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn positive_offset_crossing_sign_extend_negative_tests() {
    run_negative_auipc_test(
        AUIPC,
        Some(0x7f_fff0),
        Some(0x1000),
        AuipcPrankValues {
            is_sign_extend: Some(1),
            ..Default::default()
        },
        true,
    );
}

#[test]
fn overflow_negative_tests() {
    let (imm_low_8, imm_high_16) = split_imm_u8_limbs([3592, 219, 3]);
    run_negative_auipc_test(
        AUIPC,
        Some(256264),
        None,
        AuipcPrankValues {
            imm_low_8: Some(imm_low_8),
            imm_high_16: Some(imm_high_16),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(0),
        AuipcPrankValues {
            rd_data: Some([1, 0]),
            ..Default::default()
        },
        false,
    );
    let (imm_low_8, imm_high_16) = split_imm_u8_limbs([F::NEG_ONE.as_canonical_u32(), 1, 0]);
    run_negative_auipc_test(
        AUIPC,
        Some(255),
        None,
        AuipcPrankValues {
            imm_low_8: Some(imm_low_8),
            imm_high_16: Some(imm_high_16),
            ..Default::default()
        },
        true,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(255),
        AuipcPrankValues {
            rd_data: Some([F::NEG_ONE.as_canonical_u32(), 1]),
            imm_low_8: Some(0),
            imm_high_16: Some(0),
            ..Default::default()
        },
        true,
    );
    run_negative_auipc_test(
        AUIPC,
        Some((F::ORDER_U32 + 255) >> RV64_BYTE_BITS),
        Some(0),
        AuipcPrankValues {
            rd_data: Some([255, 0]),
            ..Default::default()
        },
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_auipc_sanity_test() {
    let initial_pc = 234567890;
    let imm = 11302451;
    let rd_data = run_auipc(initial_pc, imm);

    assert_eq!(
        rv64_u16_block_to_bytes(rd_data),
        [210, 107, 113, 186, 255, 255, 255, 255]
    );

    let rd_data = run_auipc(0x1000, 0x7f_fff0);
    assert_eq!(
        rv64_u16_block_to_bytes(rd_data),
        [0, 0, 0, 0x80, 0, 0, 0, 0]
    );

    let rd_data = run_auipc(0x2000, 0xff_fff0);
    assert_eq!(
        rv64_u16_block_to_bytes(rd_data),
        [0, 0x10, 0, 0, 0, 0, 0, 0]
    );
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64AuipcExecutor, Rv64AuipcAir, Rv64AuipcChipGpu, Rv64AuipcChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AuipcChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_auipc_tracegen() {
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(openvm_circuit::arch::testing::default_bitwise_lookup_bus());
    let mut rng = create_seeded_rng();
    let mut harness = create_cuda_harness(&tester);

    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            AUIPC,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64RdWriteAdapterRecord,
        &'a mut Rv64AuipcCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64RdWriteAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_auipc_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let auipc = |rd: usize, immediate: usize| {
        Instruction::<F>::from_usize(
            AUIPC.global_opcode(),
            [reg(rd), 0, immediate, RV64_REGISTER_AS as usize, 0],
        )
    };
    let pc_base = 0x1000;
    let instructions = [
        auipc(1, 0x7f_fff0),
        auipc(2, 0x80_0000),
        auipc(3, 0),
        auipc(1, 1),
        auipc(4, 0xff_fff0),
        auipc(5, 0x12_3456),
        auipc(6, 0x80_0001),
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::new_without_debug_infos(&instructions, pc_base);
    let exe = VmExe::new(program.clone()).with_pc_start(pc_base);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config.clone())
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(16, 16))
        .unwrap();

    // The first result is 0x8000_0000 with zero upper limbs. It must not be
    // sign-extended from the low word's sign bit.
    assert_eq!(execution.transcript.memory_log[0].value, [0, 0x8000, 0, 0]);
    // The second result is negative in the full u64 AUIPC semantics.
    assert_eq!(
        execution.transcript.memory_log[1].value,
        [0x1004, 0x8000, 0xffff, 0xffff]
    );

    let mut tester = GpuChipTestBuilder::default();
    let initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    tester.memory.memory = TracingMemory::from_image(initial_image);
    let device_ctx = tester.range_checker().device_ctx.clone();
    let hasher_chip = tester.memory.hasher_chip.clone().unwrap();
    tester.memory.inventory = MemoryInventoryGPU::new(
        tester.memory.config.clone(),
        hasher_chip,
        device_ctx.clone(),
    );
    tester
        .memory
        .inventory
        .set_initial_memory(&tester.memory.memory.data().memory);
    let mut harness = create_cuda_harness(&tester);
    for (instruction_index, instruction) in instructions[..7].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc_base + instruction_index as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64RdWriteAdapterRecord,
        &'a mut Rv64AuipcCoreRecord,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64RdWriteAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(AUIPC.global_opcode()).len(), 7);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let raw_count = |count: &F| {
        const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<u32>()) };
        // CUDA kernels atomically update this shared field-typed buffer as raw u32 counters.
        unsafe { *(std::ptr::from_ref(count).cast::<u32>()) }
    };
    assert_eq!(replay_counts.iter().map(raw_count).sum::<u32>(), 7 * 9);

    let run_corrupt = |corrupt_program: &Program<F>,
                       transcript: RvrPreflightTranscript,
                       endpoint: RvrPreflightEndpoint,
                       expected_error: u32,
                       expected_lookup_count: u32| {
        let corrupt_range_checker = Arc::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
                default_var_range_checker_bus(),
                device_ctx.clone(),
            ),
        );
        let d_corrupt_program =
            GpuRvrProgram::upload(corrupt_program, &memory_config, device_ctx).unwrap();
        let (d_corrupt, d_corrupt_plan) = d_corrupt_program
            .upload_transcript(&transcript, endpoint)
            .unwrap();
        Rv64AuipcChipGpu::new(corrupt_range_checker.clone(), tester.timestamp_max_bits())
            .generate_proving_ctx_from_rvr(&d_corrupt_program, &d_corrupt, &d_corrupt_plan)
            .unwrap();
        assert_eq!(d_corrupt.error_code().unwrap(), expected_error);
        assert_eq!(
            corrupt_range_checker
                .count
                .to_host_on(device_ctx)
                .unwrap()
                .iter()
                .map(raw_count)
                .sum::<u32>(),
            expected_lookup_count,
            "a rejected row must not update the shared lookup histogram"
        );
    };
    let transcript = || RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };

    let mut result_corrupt = transcript();
    result_corrupt.memory_log.last_mut().unwrap().value[3] = 0;
    run_corrupt(
        &program,
        result_corrupt,
        RvrPreflightEndpoint::Terminated,
        197,
        6 * 9,
    );

    let mut bound_instructions = instructions.clone();
    bound_instructions[6] = auipc(6, 1 << 24);
    run_corrupt(
        &Program::new_without_debug_infos(&bound_instructions, pc_base),
        transcript(),
        RvrPreflightEndpoint::Terminated,
        194,
        6 * 9,
    );

    let mut x0_instructions = instructions.clone();
    x0_instructions[6] = auipc(0, 0x80_0001);
    run_corrupt(
        &Program::new_without_debug_infos(&x0_instructions, pc_base),
        transcript(),
        RvrPreflightEndpoint::Terminated,
        194,
        6 * 9,
    );

    let boundary_pc = (1u32 << PC_BITS) - 4;
    let boundary_program = Program::new_without_debug_infos(&[auipc(1, 0)], boundary_pc);
    let mut boundary_from = execution.transcript.program_log[0];
    boundary_from.pc = boundary_pc;
    let mut boundary_to = boundary_from;
    boundary_to.timestamp += 1;
    let mut boundary_write = execution.transcript.memory_log[0];
    boundary_write.timestamp = boundary_from.timestamp;
    boundary_write.value = [0, 0, 0, 0];
    let boundary_transcript = RvrPreflightTranscript {
        program_log: vec![boundary_from, boundary_to],
        memory_log: vec![boundary_write],
        initial_write_log: execution.transcript.initial_write_log[..1].to_vec(),
    };
    run_corrupt(
        &boundary_program,
        boundary_transcript,
        RvrPreflightEndpoint::Suspended {
            resume_pc: boundary_pc,
            final_timestamp: boundary_to.timestamp,
        },
        192,
        0,
    );

    // The shared postflight derives predecessor indexes, so corrupt that
    // derived input directly to exercise the AUIPC kernel's fail-closed check.
    let mut corrupt_predecessors = d_transcript.memory_predecessors_host().unwrap();
    *corrupt_predecessors.last_mut().unwrap() = u32::MAX;
    let d_corrupt_predecessors = corrupt_predecessors.to_device_on(device_ctx).unwrap();
    let predecessor_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let trace_width = Rv64AuipcCoreCols::<F>::width() + Rv64RdWriteAdapterCols::<F>::width();
    let d_predecessor_trace = DeviceMatrix::<F>::with_capacity_on(8, trace_width, device_ctx);
    let step_range = d_replay_plan.opcode_range(AUIPC.global_opcode());
    unsafe {
        crate::cuda_abi::auipc_cuda::replay_tracegen(
            d_predecessor_trace.buffer(),
            8,
            d_program.instructions(),
            d_program.pc_base(),
            d_transcript.program_log(),
            d_transcript.memory_log(),
            d_transcript.initial_write_log(),
            d_corrupt_predecessors.view(),
            d_replay_plan.steps(),
            step_range.start,
            step_range.len(),
            d_transcript.error_ptr(),
            AUIPC.global_opcode().as_usize() as u32,
            RV64_REGISTER_AS,
            &predecessor_range_checker.count,
            tester.timestamp_max_bits() as u32,
            device_ctx.stream.as_raw(),
        )
        .unwrap();
    }
    assert_eq!(d_transcript.error_code().unwrap(), 198);
    assert_eq!(
        predecessor_range_checker
            .count
            .to_host_on(device_ctx)
            .unwrap()
            .iter()
            .map(raw_count)
            .sum::<u32>(),
        6 * 9,
        "the row with an invalid predecessor must not update the histogram"
    );

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_ctx =
        Rv64AuipcChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits())
            .generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace =
        <Rv64AuipcChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
        .common_main;
    let replay_trace = transport_matrix_d2h_row_major(&replay_ctx.common_main, device_ctx).unwrap();
    assert_eq!(expected_trace, replay_trace);
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR AUIPC transcript replay proof failed");
}
