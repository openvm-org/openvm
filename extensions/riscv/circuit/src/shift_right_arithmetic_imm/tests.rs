use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
    BLOCK_FE_WIDTH,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::ShiftImmOpcode;
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
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
    openvm_circuit_primitives::Chip,
    openvm_cpu_backend::CpuBackend,
    openvm_cuda_backend::data_transporter::assert_eq_host_and_device_matrix_col_maj,
    openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
    openvm_instructions::{
        exe::{SparseMemoryImage, VmExe},
        instruction::Instruction,
        program::Program,
        riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
        SystemOpcode,
    },
    openvm_stark_backend::prover::ColMajorMatrix,
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::Rv64BaseAluImmU16AdapterRecord, Rv64ShiftRightArithmeticImmChipGpu,
        ShiftRightArithmeticImmCoreRecord,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{
    Rv64ShiftRightArithmeticImmAir, Rv64ShiftRightArithmeticImmChip,
    Rv64ShiftRightArithmeticImmExecutor, ShiftRightArithmeticImmCoreAir,
    ShiftRightArithmeticImmCoreCols, ShiftRightArithmeticImmFiller,
};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, U16_BITS,
    },
    test_utils::rv64_rand_write_register_or_imm,
};

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv64ShiftRightArithmeticImmExecutor,
    Rv64ShiftRightArithmeticImmAir,
    Rv64ShiftRightArithmeticImmChip<F>,
>;

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let air = Rv64ShiftRightArithmeticImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftRightArithmeticImmCoreAir::new(range_checker.bus(), ShiftImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64ShiftRightArithmeticImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        ShiftImmOpcode::CLASS_OFFSET,
    );
    let chip = Rv64ShiftRightArithmeticImmChip::new(
        ShiftRightArithmeticImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker),
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, 32)
}

#[test]
fn rv64_shift_right_arithmetic_immediate_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for source in [0x0123_4567_89ab_cdefu64, 0x8123_4567_89ab_cdef, u64::MAX] {
        for shamt in [0usize, 1, 15, 16, 31, 32, 63] {
            let (instruction, rd) = rv64_rand_write_register_or_imm(
                &mut tester,
                source.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(shamt),
                ShiftImmOpcode::SRAI.global_opcode().as_usize(),
                &mut rng,
            );
            tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

            let expected = ((source as i64) >> shamt) as u64;
            assert_eq!(
                expected.to_le_bytes().map(F::from_u8),
                tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
            );
        }
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[test]
fn rv64_shift_right_arithmetic_immediate_marker_negative() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);
    let (instruction, _) = rv64_rand_write_register_or_imm(
        &mut tester,
        0x8123_4567_89ab_cdefu64.to_le_bytes(),
        [0; RV64_REGISTER_NUM_LIMBS],
        Some(1),
        ShiftImmOpcode::SRAI.global_opcode().as_usize(),
        &mut rng,
    );
    tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut ShiftRightArithmeticImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.bit_shift_marker = [F::ZERO; U16_BITS];
        cols.bit_shift_marker[2] = F::ONE;
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize()
        .simple_test()
        .expect_err("altered shift marker should fail");
}

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64ShiftRightArithmeticImmExecutor,
    Rv64ShiftRightArithmeticImmAir,
    Rv64ShiftRightArithmeticImmChipGpu,
    Rv64ShiftRightArithmeticImmChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_checker = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));
    let air = Rv64ShiftRightArithmeticImmAir::new(
        Rv64BaseAluImmU16AdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        ShiftRightArithmeticImmCoreAir::new(range_checker.bus(), ShiftImmOpcode::CLASS_OFFSET),
    );
    let executor = Rv64ShiftRightArithmeticImmExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor::new(),
        ShiftImmOpcode::CLASS_OFFSET,
    );
    let cpu_chip = Rv64ShiftRightArithmeticImmChip::new(
        ShiftRightArithmeticImmFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64ShiftRightArithmeticImmChipGpu::new(
        tester.range_checker(),
        tester.timestamp_max_bits(),
    );
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 32)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_shift_right_arithmetic_immediate_boundaries_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);

    for (source, shamt) in [
        (0x0123_4567_89ab_cdefu64, 0usize),
        (0x8123_4567_89ab_cdef, 1),
        (u64::MAX, 15),
        (0x8123_4567_89ab_cdef, 16),
        (0x0123_4567_89ab_cdef, 31),
        (0x8123_4567_89ab_cdef, 32),
        (u64::MAX, 63),
    ] {
        let (instruction, _) = rv64_rand_write_register_or_imm(
            &mut tester,
            source.to_le_bytes(),
            [0; RV64_REGISTER_NUM_LIMBS],
            Some(shamt),
            ShiftImmOpcode::SRAI.global_opcode().as_usize(),
            &mut rng,
        );
        tester.execute(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instruction,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut ShiftRightArithmeticImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
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
fn test_cuda_shift_right_arithmetic_immediate_tracegen_from_rvr_transcript() {
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let instruction = |rd: usize, rs1: usize, shamt: usize| {
        Instruction::<F>::from_usize(
            ShiftImmOpcode::SRAI.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                shamt,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        )
    };
    let instructions = [
        instruction(2, 1, 0),
        instruction(3, 1, 1),
        instruction(4, 1, 15),
        instruction(5, 1, 16),
        instruction(6, 1, 31),
        instruction(7, 1, 32),
        instruction(8, 1, 47),
        instruction(9, 1, 48),
        instruction(10, 1, 63),
        instruction(12, 11, 63),
        instruction(13, 0, 63),
        instruction(1, 1, 1),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let source = 0x8123_4567_89ab_cdefu64.to_le_bytes();
    let mut init_memory: SparseMemoryImage = source
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
        .collect();
    init_memory.extend(
        0x7fff_ffff_ffff_ffffu64
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(11) + offset) as u32), byte)),
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
        .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(13, 24))
        .unwrap();

    let mut tester = GpuChipTestBuilder::default();
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
    for (pc, instruction) in instructions[..12].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            pc as u32 * 4,
        );
    }
    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut ShiftRightArithmeticImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(
        d_replay_plan
            .opcode_range(ShiftImmOpcode::SRAI.global_opcode())
            .len(),
        12
    );
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let cross_limb_write_timestamp = corrupt_transcript.program_log[6].timestamp + 1;
    let cross_limb_write = corrupt_transcript
        .memory_log
        .iter_mut()
        .find(|event| event.timestamp == cross_limb_write_timestamp)
        .unwrap();
    cross_limb_write.value[1] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_chip =
        Rv64ShiftRightArithmeticImmChipGpu::new(corrupt_range_checker, tester.timestamp_max_bits());
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 68);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_chip = Rv64ShiftRightArithmeticImmChipGpu::new(
        legacy_range_checker.clone(),
        tester.timestamp_max_bits(),
    );
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    assert_eq!(
        replay_counts,
        legacy_range_checker.count.to_host_on(device_ctx).unwrap()
    );

    let expected_trace = <Rv64ShiftRightArithmeticImmChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<openvm_cuda_backend::prelude::SC>,
    >>::generate_proving_ctx(&harness.cpu_chip, harness.matrix_arena)
    .common_main;
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &replay_ctx.common_main, device_ctx);
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR SRAI transcript replay proof failed");
}

mod word {
    use openvm_circuit::arch::testing::{TestBuilder, TestChipHarness, VmChipTestBuilder};
    use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
    use openvm_riscv_transpiler::ShiftWImmOpcode;
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
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
        openvm_circuit_primitives::Chip,
        openvm_cpu_backend::CpuBackend,
        openvm_cuda_backend::{
            data_transporter::assert_eq_host_and_device_matrix_col_maj, prelude::SC,
        },
        openvm_cuda_common::{copy::MemCopyD2H, stream::device_synchronize},
        openvm_instructions::{
            exe::{SparseMemoryImage, VmExe},
            instruction::Instruction,
            program::Program,
            riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
            SystemOpcode,
        },
        openvm_stark_backend::prover::ColMajorMatrix,
    };
    #[cfg(feature = "cuda")]
    use {
        crate::{
            adapters::{Rv64BaseAluWImmU16AdapterRecord, RV64_WORD_U16_LIMBS, U16_BITS},
            Rv64ShiftWRightArithmeticImmChipGpu, ShiftRightArithmeticImmCoreRecord,
        },
        openvm_circuit::arch::{
            testing::{GpuChipTestBuilder, GpuTestChipHarness},
            EmptyAdapterCoreLayout,
        },
        openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
        std::sync::Arc,
    };

    use crate::{
        adapters::{
            Rv64BaseAluWImmU16AdapterAir, Rv64BaseAluWImmU16AdapterExecutor,
            Rv64BaseAluWImmU16AdapterFiller,
        },
        shift_right_arithmetic_imm::{
            Rv64ShiftWRightArithmeticImmAir, Rv64ShiftWRightArithmeticImmChip,
            Rv64ShiftWRightArithmeticImmExecutor, ShiftRightArithmeticImmCoreAir,
            ShiftRightArithmeticImmFiller,
        },
        test_utils::rv64_rand_write_register_or_imm,
    };

    type F = BabyBear;
    type Harness = TestChipHarness<
        F,
        Rv64ShiftWRightArithmeticImmExecutor,
        Rv64ShiftWRightArithmeticImmAir,
        Rv64ShiftWRightArithmeticImmChip<F>,
    >;

    fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
        let range_checker = tester.range_checker();
        let air = Rv64ShiftWRightArithmeticImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftRightArithmeticImmCoreAir::new(range_checker.bus(), ShiftWImmOpcode::CLASS_OFFSET),
        );
        let executor = Rv64ShiftWRightArithmeticImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
        );
        let chip = Rv64ShiftWRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker,
            ),
            tester.memory_helper(),
        );
        Harness::with_capacity(executor, air, chip, 16)
    }

    #[test]
    fn rv64_shift_w_right_arithmetic_immediate_boundaries() {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let mut harness = create_harness(&tester);

        for source in [0xa5a5_a5a5_1234_5678u64, 0x5a5a_5a5a_8765_4321] {
            for shamt in [0usize, 1, 15, 16, 31] {
                let (instruction, rd) = rv64_rand_write_register_or_imm(
                    &mut tester,
                    source.to_le_bytes(),
                    [0; RV64_REGISTER_NUM_LIMBS],
                    Some(shamt),
                    ShiftWImmOpcode::SRAIW.global_opcode().as_usize(),
                    &mut rng,
                );
                tester.execute(&mut harness.executor, &mut harness.arena, &instruction);

                let expected = ((source as u32 as i32) >> shamt) as i64 as u64;
                assert_eq!(
                    expected.to_le_bytes().map(F::from_u8),
                    tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd),
                    "SRAIW source={source:#018x} shamt={shamt}",
                );
            }
        }

        tester
            .build()
            .load(harness)
            .finalize()
            .simple_test()
            .expect("verification failed");
    }

    // ////////////////////////////////////////////////////////////////////////////////////
    //  CUDA TESTS
    //
    //  Ensure GPU tracegen is equivalent to CPU tracegen.
    // ////////////////////////////////////////////////////////////////////////////////////

    #[cfg(feature = "cuda")]
    type GpuHarness = GpuTestChipHarness<
        F,
        Rv64ShiftWRightArithmeticImmExecutor,
        Rv64ShiftWRightArithmeticImmAir,
        Rv64ShiftWRightArithmeticImmChipGpu,
        Rv64ShiftWRightArithmeticImmChip<F>,
    >;

    #[cfg(feature = "cuda")]
    fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
        let range_checker = Arc::new(VariableRangeCheckerChip::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
        ));
        let air = Rv64ShiftWRightArithmeticImmAir::new(
            Rv64BaseAluWImmU16AdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                range_checker.bus(),
            ),
            ShiftRightArithmeticImmCoreAir::new(range_checker.bus(), ShiftWImmOpcode::CLASS_OFFSET),
        );
        let executor = Rv64ShiftWRightArithmeticImmExecutor::new(
            Rv64BaseAluWImmU16AdapterExecutor,
            ShiftWImmOpcode::CLASS_OFFSET,
        );
        let cpu_chip = Rv64ShiftWRightArithmeticImmChip::new(
            ShiftRightArithmeticImmFiller::new(
                Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
                range_checker,
            ),
            tester.dummy_memory_helper(),
        );
        let gpu_chip = Rv64ShiftWRightArithmeticImmChipGpu::new(
            tester.range_checker(),
            tester.timestamp_max_bits(),
        );

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 16)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_shift_w_right_arithmetic_immediate_tracegen() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        let mut harness = create_cuda_harness(&tester);

        for (source, shamt) in [
            (0xa5a5_a5a5_7fff_ffffu64, 0usize),
            (0x5a5a_5a5a_8000_0000, 1),
            (0xa5a5_a5a5_8000_0001, 15),
            (0x5a5a_5a5a_ffff_ffff, 16),
            (0xa5a5_a5a5_8000_0000, 31),
        ] {
            let (instruction, _) = rv64_rand_write_register_or_imm(
                &mut tester,
                source.to_le_bytes(),
                [0; RV64_REGISTER_NUM_LIMBS],
                Some(shamt),
                ShiftWImmOpcode::SRAIW.global_opcode().as_usize(),
                &mut rng,
            );
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }

        type Record<'a> = (
            &'a mut Rv64BaseAluWImmU16AdapterRecord,
            &'a mut ShiftRightArithmeticImmCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv64BaseAluWImmU16AdapterExecutor>::new(),
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
    fn test_cuda_shift_w_right_arithmetic_immediate_tracegen_from_rvr_transcript() {
        let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
        let instruction = |rd: usize, rs1: usize, shamt: usize| {
            Instruction::<F>::from_usize(
                ShiftWImmOpcode::SRAIW.global_opcode(),
                [
                    reg(rd),
                    reg(rs1),
                    shamt,
                    RV64_REGISTER_AS as usize,
                    RV64_IMM_AS as usize,
                ],
            )
        };
        let instructions = [
            instruction(2, 1, 0),
            instruction(3, 1, 1),
            instruction(4, 1, 15),
            instruction(5, 1, 16),
            instruction(6, 1, 31),
            instruction(7, 11, 1),
            instruction(8, 0, 31),
            instruction(9, 11, 31),
            instruction(1, 1, 1),
            Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
        ];
        let program = Program::from_instructions(&instructions);
        let mut init_memory: SparseMemoryImage = 0xa5a5_a5a5_8000_0001u64
            .to_le_bytes()
            .into_iter()
            .enumerate()
            .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(1) + offset) as u32), byte))
            .collect();
        init_memory.extend(
            0x5a5a_5a5a_7fff_ffffu64
                .to_le_bytes()
                .into_iter()
                .enumerate()
                .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(11) + offset) as u32), byte)),
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
            .execute(Vec::<Vec<u8>>::new(), RvrPreflightLimits::new(10, 18))
            .unwrap();

        let mut tester = GpuChipTestBuilder::default();
        let mut initial_image =
            GuestMemory::new(AddressMap::from_mem_config(&tester.memory.config));
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
        for (pc, instruction) in instructions[..9].iter().enumerate() {
            tester.execute_with_pc(
                &mut harness.executor,
                &mut harness.dense_arena,
                instruction,
                pc as u32 * 4,
            );
        }
        type Record<'a> = (
            &'a mut Rv64BaseAluWImmU16AdapterRecord,
            &'a mut ShiftRightArithmeticImmCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        );
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(
                &mut harness.matrix_arena,
                EmptyAdapterCoreLayout::<F, Rv64BaseAluWImmU16AdapterExecutor>::new(),
            );

        let range_checker = tester.range_checker();
        let device_ctx = &range_checker.device_ctx;
        let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
        let (d_transcript, d_replay_plan) = d_program
            .upload_transcript(&execution.transcript, execution.endpoint)
            .unwrap();
        assert_eq!(
            d_replay_plan
                .opcode_range(ShiftWImmOpcode::SRAIW.global_opcode())
                .len(),
            9
        );
        let replay_ctx = harness
            .gpu_chip
            .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
            .unwrap();
        assert_eq!(d_transcript.error_code().unwrap(), 0);
        let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();

        let mut corrupt_transcript = RvrPreflightTranscript {
            program_log: execution.transcript.program_log.clone(),
            memory_log: execution.transcript.memory_log.clone(),
            initial_write_log: execution.transcript.initial_write_log.clone(),
        };
        let sign_extended_write_timestamp = corrupt_transcript.program_log[2].timestamp + 1;
        let sign_extended_write = corrupt_transcript
            .memory_log
            .iter_mut()
            .find(|event| event.timestamp == sign_extended_write_timestamp)
            .unwrap();
        sign_extended_write.value[2] ^= 1;
        let (d_corrupt, d_corrupt_plan) = d_program
            .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
            .unwrap();
        let corrupt_range_checker = Arc::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
                openvm_circuit::arch::testing::default_var_range_checker_bus(),
                device_ctx.clone(),
            ),
        );
        let corrupt_chip = Rv64ShiftWRightArithmeticImmChipGpu::new(
            corrupt_range_checker,
            tester.timestamp_max_bits(),
        );
        corrupt_chip
            .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
            .unwrap();
        assert_eq!(d_corrupt.error_code().unwrap(), 88);

        let legacy_range_checker = Arc::new(
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
                openvm_circuit::arch::testing::default_var_range_checker_bus(),
                device_ctx.clone(),
            ),
        );
        let legacy_chip = Rv64ShiftWRightArithmeticImmChipGpu::new(
            legacy_range_checker.clone(),
            tester.timestamp_max_bits(),
        );
        let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
        assert_eq!(
            replay_counts,
            legacy_range_checker.count.to_host_on(device_ctx).unwrap()
        );

        let expected_trace = <Rv64ShiftWRightArithmeticImmChip<F> as Chip<
            MatrixRecordArena<F>,
            CpuBackend<SC>,
        >>::generate_proving_ctx(
            &harness.cpu_chip, harness.matrix_arena
        )
        .common_main;
        let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
        device_synchronize().unwrap();
        assert_eq_host_and_device_matrix_col_maj(
            &expected_trace,
            &replay_ctx.common_main,
            device_ctx,
        );
        assert_eq_host_and_device_matrix_col_maj(
            &expected_trace,
            &legacy_ctx.common_main,
            device_ctx,
        );

        tester
            .build()
            .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
            .finalize()
            .simple_test()
            .expect("RVR SRAIW transcript replay proof failed");
    }
}
