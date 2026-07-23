use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluOpcode::{self, *};
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
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::GpuRvrProgram, RvrPreflightEndpoint, RvrPreflightLimits,
                RvrPreflightTranscript,
            },
            MatrixRecordArena, VmExecutor,
        },
        system::{
            cuda::memory::MemoryInventoryGPU,
            memory::online::{AddressMap, GuestMemory, TracingMemory},
        },
        utils::test_system_config,
    },
    openvm_circuit_primitives::{
        bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU,
        Chip,
    },
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::{
        data_transporter::{
            assert_eq_host_and_device_matrix_col_maj, transport_matrix_d2h_row_major,
        },
        prelude::SC,
    },
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        instruction::Instruction,
        program::Program,
        riscv::RV64_REGISTER_AS,
        SystemOpcode,
    },
    openvm_stark_backend::{p3_field::PrimeField32, prover::ColMajorMatrix},
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BaseAluRegAdapterRecord, BitwiseLogicCoreRecord, Rv64BitwiseLogicChipGpu,
    },
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{
    core::run_bitwise_logic, BitwiseLogicCoreAir, Rv64BitwiseLogicChip, Rv64BitwiseLogicExecutor,
};
use crate::{
    adapters::{
        Rv64BaseAluRegAdapterAir, Rv64BaseAluRegAdapterExecutor, Rv64BaseAluRegAdapterFiller,
        RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    bitwise_logic::BitwiseLogicCoreCols,
    test_utils::rv64_rand_write_register_or_imm,
    BitwiseLogicFiller, Rv64BitwiseLogicAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness =
    TestChipHarness<F, Rv64BitwiseLogicExecutor, Rv64BitwiseLogicAir, Rv64BitwiseLogicChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BitwiseLogicAir,
    Rv64BitwiseLogicExecutor,
    Rv64BitwiseLogicChip<F>,
) {
    let air = Rv64BitwiseLogicAir::new(
        Rv64BaseAluRegAdapterAir::new(execution_bridge, memory_bridge),
        BitwiseLogicCoreAir::new(bitwise_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BitwiseLogicExecutor::new(
        Rv64BaseAluRegAdapterExecutor::new(),
        BaseAluOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BitwiseLogicChip::new(
        BitwiseLogicFiller::new(Rv64BaseAluRegAdapterFiller, bitwise_chip),
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
        bitwise_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BaseAluOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));

    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let a = run_bitwise_logic::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(opcode, &b, &c)
        .map(F::from_u8);
    assert_eq!(a, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(XOR, 100)]
#[test_case(OR, 100)]
#[test_case(AND, 100)]
fn rand_rv64_bitwise_logic_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write_bytes(2, 1024, [F::ONE; 8]);
    tester.write_bytes(2, 1032, [F::ONE; 8]);
    let sm_lo: [F; 8] = tester.read_bytes(2, 1024);
    let sm_hi: [F; 8] = tester.read_bytes(2, 1032);
    assert_eq!(sm_lo, [F::ONE; 8]);
    assert_eq!(sm_hi, [F::ONE; 8]);

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

#[allow(clippy::too_many_arguments)]
fn run_negative_bitwise_logic_test(
    opcode: BaseAluOpcode,
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_opcode_flags: Option<[bool; 3]>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&tester);

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
        let cols: &mut BitwiseLogicCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_xor_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_or_flag = F::from_bool(prank_opcode_flags[1]);
            cols.opcode_and_flag = F::from_bool(prank_opcode_flags[2]);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
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
fn rv64_bitwise_logic_xor_wrong_negative_test() {
    run_negative_bitwise_logic_test(
        XOR,
        [255, 255, 255, 255, 255, 255, 255, 255],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [255, 255, 255, 255, 255, 255, 255, 255],
        None,
        true,
    );
}

#[test]
fn rv64_bitwise_logic_or_wrong_negative_test() {
    run_negative_bitwise_logic_test(
        OR,
        [255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 255, 254, 255, 255, 255, 255],
        [0, 0, 0, 0, 0, 0, 0, 0],
        None,
        true,
    );
}

#[test]
fn rv64_bitwise_logic_and_wrong_negative_test() {
    run_negative_bitwise_logic_test(
        AND,
        [255, 255, 255, 255, 255, 255, 255, 255],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        None,
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_xor_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [215, 138, 49, 173, 216, 1, 0, 3];
    let result = run_bitwise_logic::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(XOR, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_or_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [247, 171, 61, 239, 217, 35, 25, 207];
    let result = run_bitwise_logic::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(OR, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_and_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [50, 171, 44, 194, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [32, 33, 12, 66, 1, 34, 25, 204];
    let result = run_bitwise_logic::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(AND, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
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
    Rv64BitwiseLogicExecutor,
    Rv64BitwiseLogicAir,
    Rv64BitwiseLogicChipGpu,
    Rv64BitwiseLogicChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BitwiseLogicChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BaseAluOpcode::XOR, 100)]
#[test_case(BaseAluOpcode::OR, 100)]
#[test_case(BaseAluOpcode::AND, 100)]
fn test_cuda_rand_bitwise_logic_tracegen(opcode: BaseAluOpcode, num_ops: usize) {
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
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluRegAdapterRecord,
        &'a mut BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegAdapterExecutor>::new(),
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
fn test_cuda_bitwise_logic_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let instruction = |opcode: BaseAluOpcode, rd: usize, rs1: usize, rs2: usize| {
        Instruction::<F>::from_usize(
            opcode.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                reg(rs2),
                RV64_REGISTER_AS as usize,
                RV64_REGISTER_AS as usize,
            ],
        )
    };
    let instructions = [
        instruction(XOR, 3, 1, 2),
        instruction(OR, 4, 1, 2),
        instruction(AND, 5, 1, 1),
        instruction(XOR, 6, 0, 1),
        instruction(OR, 1, 1, 0),
        instruction(AND, 2, 1, 2),
        instruction(XOR, 1, 1, 1),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let mut init_memory: SparseMemoryImage = 0x00ff_0f0f_aaaa_5555u64
        .to_le_bytes()
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
        .collect();
    init_memory.extend(
        0xf0f0_ffff_1234_5678u64
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(2) + offset) as u32), byte)),
    );
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let execution = VmExecutor::new(config)
        .unwrap()
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(8, 21))
        .unwrap();

    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let mut initial_image = GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
    initial_image.memory.set_from_sparse(&init_memory);
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
    for (pc, instruction) in instructions[..7].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BaseAluRegAdapterRecord,
        &'a mut BitwiseLogicCoreRecord<RV64_REGISTER_NUM_LIMBS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let bitwise_lookup = tester.bitwise_op_lookup();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.opcode_range(XOR.global_opcode()).len(), 3);
    assert_eq!(d_replay_plan.opcode_range(OR.global_opcode()).len(), 2);
    assert_eq!(d_replay_plan.opcode_range(AND.global_opcode()).len(), 2);
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_range_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let replay_bitwise_counts = bitwise_lookup.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    corrupt_transcript.memory_log[2].value[0] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_chip = Rv64BitwiseLogicChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        tester.timestamp_max_bits(),
    );
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 138);

    // On the final rd == rs1 == rs2 row, alter only the second read and make the logged result
    // arithmetically consistent with it. Expected-result checking then passes, but predecessor
    // resolution must reject the second read because the first read is its immediate predecessor.
    let mut alias_corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let alias_timestamp = alias_corrupt_transcript.program_log[6].timestamp;
    let rs1_index = alias_corrupt_transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == alias_timestamp)
        .unwrap();
    let rs2_index = rs1_index + 1;
    let write_index = rs1_index + 2;
    alias_corrupt_transcript.memory_log[rs2_index].value[0] ^= 1;
    for limb in 0..openvm_circuit::arch::BLOCK_FE_WIDTH {
        let result = alias_corrupt_transcript.memory_log[rs1_index].value[limb]
            ^ alias_corrupt_transcript.memory_log[rs2_index].value[limb];
        alias_corrupt_transcript.memory_log[write_index].value[limb] = result;
    }
    let (d_alias_corrupt, d_alias_corrupt_plan) = d_program
        .upload_transcript(&alias_corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let alias_corrupt_chip = Rv64BitwiseLogicChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        tester.timestamp_max_bits(),
    );
    alias_corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_alias_corrupt, &d_alias_corrupt_plan)
        .unwrap();
    assert_eq!(d_alias_corrupt.error_code().unwrap(), 139);

    // The bitwise AIR always emits a destination write, so replay must reject an x0 destination
    // rather than synthesize a disabled write row.
    let mut x0_instructions = instructions;
    x0_instructions[0] = instruction(XOR, 0, 1, 2);
    let x0_program = Program::from_instructions(&x0_instructions);
    let d_x0_program = GpuRvrProgram::upload(&x0_program, &memory_config, device_ctx).unwrap();
    let (d_x0, d_x0_plan) = d_x0_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    let x0_chip = Rv64BitwiseLogicChipGpu::new(
        Arc::new(VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        )),
        Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone())),
        tester.timestamp_max_bits(),
    );
    x0_chip
        .generate_proving_ctx_from_rvr(&d_x0_program, &d_x0, &d_x0_plan)
        .unwrap();
    assert_eq!(d_x0.error_code().unwrap(), 134);

    let legacy_range_checker = Arc::new(VariableRangeCheckerChipGPU::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
        device_ctx.clone(),
    ));
    let legacy_bitwise_lookup = Arc::new(BitwiseOperationLookupChipGPU::new(device_ctx.clone()));
    let legacy_chip = Rv64BitwiseLogicChipGpu::new(
        legacy_range_checker.clone(),
        legacy_bitwise_lookup.clone(),
        tester.timestamp_max_bits(),
    );
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_range_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );
    assert_eq!(
        replay_bitwise_counts,
        legacy_bitwise_lookup.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace =
        <Rv64BitwiseLogicChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
        .common_main;
    let replay_trace = transport_matrix_d2h_row_major(&replay_ctx.common_main, device_ctx).unwrap();
    let canonical_rows = |matrix: &RowMajorMatrix<F>| {
        let mut rows = (0..matrix.height())
            .map(|row| matrix.row_slice(row).unwrap().to_vec())
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row[1].as_canonical_u32());
        rows
    };
    assert_eq!(
        canonical_rows(&expected_trace),
        canonical_rows(&replay_trace)
    );
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR XOR/OR/AND transcript replay proof failed");
}
