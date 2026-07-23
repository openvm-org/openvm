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
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};
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
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    crate::Rv64IConfig,
    openvm_circuit::{
        arch::{
            rvr::{
                cuda::{GpuRvrInputError, GpuRvrProgram},
                RvrPreflightEndpoint, RvrPreflightLimits, RvrPreflightTranscript,
            },
            VirtualMachine, VmExecutor,
        },
        utils::{test_cpu_engine, test_system_config},
    },
    openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::{Program, DEFAULT_PC_STEP},
        riscv::{RV64_IMM_AS, RV64_REGISTER_AS},
        SystemOpcode,
    },
    openvm_riscv_transpiler::BranchEqualOpcode,
    openvm_stark_backend::p3_field::PrimeField32,
    std::{ffi::c_void, time::Duration},
};
#[cfg(feature = "cuda")]
use {
    crate::{
        adapters::{
            Rv64BaseAluImmU16AdapterCols, Rv64BaseAluImmU16AdapterRecord,
            Rv64BaseAluWImmU16AdapterRecord, Rv64BranchAdapterAir, Rv64BranchAdapterCols,
            Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller, Rv64BranchAdapterRecord,
            RV64_WORD_U16_LIMBS,
        },
        AddICoreRecord, BranchEqualCoreAir, BranchEqualCoreCols, BranchEqualCoreRecord,
        BranchEqualFiller, Rv64AddIChipGpu, Rv64AddIWChipGpu, Rv64BranchEqualAir,
        Rv64BranchEqualChip, Rv64BranchEqualChipGpu, Rv64BranchEqualExecutor,
    },
    openvm_circuit::arch::{
        testing::{default_var_range_checker_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout, MatrixRecordArena,
    },
    openvm_circuit_primitives::{var_range::VariableRangeCheckerChip, Chip},
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
        stream::{cudaStream_t, device_synchronize},
    },
    openvm_stark_backend::prover::ColMajorMatrix,
    std::sync::Arc,
};

use super::{AddICoreAir, Rv64AddIChip, Rv64AddIExecutor, Rv64AddIWChip, Rv64AddIWExecutor};
use crate::{
    adapters::{
        Rv64BaseAluImmU16AdapterAir, Rv64BaseAluImmU16AdapterExecutor,
        Rv64BaseAluImmU16AdapterFiller, Rv64BaseAluWImmU16AdapterAir,
        Rv64BaseAluWImmU16AdapterExecutor, Rv64BaseAluWImmU16AdapterFiller,
        RV64_REGISTER_NUM_LIMBS, U16_BITS,
    },
    addi::AddICoreCols,
    test_utils::{generate_rv64_is_type_immediate, rv64_rand_write_register_or_imm},
    AddIFiller, Rv64AddIAir, Rv64AddIWAir,
};

const MAX_INS_CAPACITY: usize = 128;
const NONCANONICAL_ZERO: [u32; BLOCK_FE_WIDTH] = [
    1 << U16_BITS,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
    (1 << U16_BITS) - 1,
];
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChip<F>>;
type WHarness = TestChipHarness<F, Rv64AddIWExecutor, Rv64AddIWAir, Rv64AddIWChip<F>>;
#[cfg(feature = "cuda")]
type GpuWHarness =
    GpuTestChipHarness<F, Rv64AddIWExecutor, Rv64AddIWAir, Rv64AddIWChipGpu, Rv64AddIWChip<F>>;
#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64AddIExecutor, Rv64AddIAir, Rv64AddIChipGpu, Rv64AddIChip<F>>;
#[cfg(feature = "cuda")]
type GpuBranchHarness = GpuTestChipHarness<
    F,
    Rv64BranchEqualExecutor,
    Rv64BranchEqualAir,
    Rv64BranchEqualChipGpu,
    Rv64BranchEqualChip<F>,
>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: cudaStream_t) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(milliseconds: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct BenchmarkCudaEvent(*mut c_void);

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl BenchmarkCudaEvent {
    fn new() -> Self {
        let mut event = std::ptr::null_mut();
        assert_eq!(unsafe { cudaEventCreate(&mut event) }, 0);
        Self(event)
    }

    fn record(&self, stream: cudaStream_t) {
        assert_eq!(unsafe { cudaEventRecord(self.0, stream) }, 0);
    }

    fn synchronize(&self) {
        assert_eq!(unsafe { cudaEventSynchronize(self.0) }, 0);
    }

    fn elapsed_ms(&self, end: &Self) -> f32 {
        let mut milliseconds = 0.0;
        assert_eq!(
            unsafe { cudaEventElapsedTime(&mut milliseconds, self.0, end.0) },
            0
        );
        milliseconds
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl Drop for BenchmarkCudaEvent {
    fn drop(&mut self) {
        assert_eq!(unsafe { cudaEventDestroy(self.0) }, 0);
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn benchmark_cuda_launches(stream: cudaStream_t, launches: usize, mut launch: impl FnMut()) -> f64 {
    let start = BenchmarkCudaEvent::new();
    let end = BenchmarkCudaEvent::new();
    start.record(stream);
    for _ in 0..launches {
        launch();
    }
    end.record(stream);
    end.synchronize();
    f64::from(start.elapsed_ms(&end)) / launches as f64
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn duration_percentiles(mut values: Vec<Duration>) -> (Duration, Duration, Duration) {
    values.sort_unstable();
    let last = values.len() - 1;
    (values[last / 10], values[last / 2], values[last * 9 / 10])
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn f64_percentiles(mut values: Vec<f64>) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let last = values.len() - 1;
    (values[last / 10], values[last / 2], values[last * 9 / 10])
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIAir, Rv64AddIExecutor, Rv64AddIChip<F>) {
    let air = Rv64AddIAir::new(
        Rv64BaseAluImmU16AdapterAir::new(execution_bridge, memory_bridge),
        AddICoreAir::new(
            range_checker_chip.bus(),
            BaseAluImmOpcode::CLASS_OFFSET,
            BaseAluImmOpcode::ADDI as usize,
        ),
    );
    let executor = Rv64AddIExecutor::new(
        Rv64BaseAluImmU16AdapterExecutor,
        BaseAluImmOpcode::CLASS_OFFSET,
        BaseAluImmOpcode::ADDI as usize,
    );
    let chip = Rv64AddIChip::new(
        AddIFiller::new(Rv64BaseAluImmU16AdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn create_w_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddIWAir, Rv64AddIWExecutor, Rv64AddIWChip<F>) {
    let air = Rv64AddIWAir::new(
        Rv64BaseAluWImmU16AdapterAir::new(execution_bridge, memory_bridge, range_checker.bus()),
        AddICoreAir::new(
            range_checker.bus(),
            BaseAluWImmOpcode::CLASS_OFFSET,
            BaseAluWImmOpcode::ADDIW as usize,
        ),
    );
    let executor = Rv64AddIWExecutor::new(
        Rv64BaseAluWImmU16AdapterExecutor,
        BaseAluWImmOpcode::CLASS_OFFSET,
        BaseAluWImmOpcode::ADDIW as usize,
    );
    let chip = Rv64AddIWChip::new(
        AddIFiller::new(
            Rv64BaseAluWImmU16AdapterFiller::new(range_checker.clone()),
            range_checker,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_w_harness(tester: &VmChipTestBuilder<F>) -> WHarness {
    let (air, executor, chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
    );
    WHarness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_cuda_w_harness(tester: &GpuChipTestBuilder) -> GpuWHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_w_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddIWChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, 8)
}

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    create_cuda_harness_with_capacity(tester, 64)
}

#[cfg(feature = "cuda")]
fn create_cuda_harness_with_capacity(tester: &GpuChipTestBuilder, capacity: usize) -> GpuHarness {
    let dummy_range_checker = Arc::new(VariableRangeCheckerChip::new(
        default_var_range_checker_bus(),
    ));
    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddIChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, capacity)
}

#[cfg(feature = "cuda")]
fn create_cuda_branch_harness_with_capacity(
    tester: &GpuChipTestBuilder,
    capacity: usize,
) -> GpuBranchHarness {
    let air = Rv64BranchEqualAir::new(
        Rv64BranchAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
    );
    let executor = Rv64BranchEqualExecutor::new(
        Rv64BranchAdapterExecutor,
        BranchEqualOpcode::CLASS_OFFSET,
        DEFAULT_PC_STEP,
    );
    let cpu_chip = Rv64BranchEqualChip::new(
        BranchEqualFiller::new(
            Rv64BranchAdapterFiller,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64BranchEqualChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, capacity)
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let (imm, c) = if let Some(c) = c {
        ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
    } else {
        generate_rv64_is_type_immediate(rng)
    };

    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        Some(imm),
        BaseAluImmOpcode::ADDI.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let rs1 = u64::from_le_bytes(b);
    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = rs1.wrapping_add(signed_imm as i64 as u64);
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

fn set_and_execute_w<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    rs1: u64,
    imm: usize,
) {
    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        rs1.to_le_bytes(),
        (imm as u64).to_le_bytes(),
        Some(imm),
        BaseAluWImmOpcode::ADDIW.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let signed_imm = ((imm as u32) << 20) as i32 >> 20;
    let expected = (rs1 as u32).wrapping_add(signed_imm as u32) as i32 as i64 as u64;
    assert_eq!(
        expected.to_le_bytes().map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_rv64_addi_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    for _ in 0..100 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            None,
            None,
        );
    }

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv64_addiw_boundaries_and_sign_extension() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            rs1,
            imm,
        );
    }

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_addiw_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_w_harness(&tester);

    for (rs1, imm) in [
        (0x0000_0000_0000_0000u64, 0x00_0000usize),
        (0x0000_0000_0000_0000, 0x00_07ff),
        (0x0000_0000_0000_0000, 0x00_fff800),
        (0x0000_0000_0000_0000, 0x00_ffffff),
        (0x0000_0000_7fff_ffff, 0x00_000001),
    ] {
        set_and_execute_w(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            rs1,
            imm,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BaseAluWImmU16AdapterRecord,
        &'a mut AddICoreRecord<RV64_WORD_U16_LIMBS>,
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
        .expect("verification failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_cuda_addi_tracegen_from_rvr_transcript() {
    const ITERATIONS: usize = 32;
    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let addi = |rd: usize, rs1: usize, immediate: usize| {
        Instruction::<F>::from_usize(
            BaseAluImmOpcode::ADDI.global_opcode(),
            [
                reg(rd),
                reg(rs1),
                immediate,
                RV64_REGISTER_AS as usize,
                RV64_IMM_AS as usize,
            ],
        )
    };
    let instructions = vec![
        addi(1, 0, 0),
        addi(2, 0, ITERATIONS),
        Instruction::from_isize(
            BranchEqualOpcode::BEQ.global_opcode(),
            reg(1) as isize,
            reg(0) as isize,
            8,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        addi(4, 0, 123),
        addi(1, 1, 1),
        Instruction::from_isize(
            BranchEqualOpcode::BEQ.global_opcode(),
            reg(0) as isize,
            reg(0) as isize,
            4,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        Instruction::from_isize(
            BranchEqualOpcode::BNE.global_opcode(),
            reg(1) as isize,
            reg(2) as isize,
            -8,
            RV64_REGISTER_AS as isize,
            RV64_REGISTER_AS as isize,
        ),
        addi(3, 1, 0xff_ffff),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::from(program.clone());
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let executor = VmExecutor::new(config).unwrap();
    let execution = executor
        .rvr_preflight_instance(&exe, None)
        .unwrap()
        .execute(
            Vec::<Vec<u8>>::new(),
            RvrPreflightLimits::new(3 * ITERATIONS + 5, 6 * ITERATIONS + 8),
        )
        .unwrap();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_cuda_harness(&tester);
    let mut branch_harness = create_cuda_branch_harness_with_capacity(&tester, 2 * ITERATIONS + 1);
    for (pc, instruction) in instructions[..2].iter().enumerate() {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            instruction,
            (pc as u32) * 4,
        );
    }
    tester.execute_with_pc(
        &mut branch_harness.executor,
        &mut branch_harness.dense_arena,
        &instructions[2],
        8,
    );
    for _ in 0..ITERATIONS {
        tester.execute_with_pc(
            &mut harness.executor,
            &mut harness.dense_arena,
            &instructions[4],
            16,
        );
        tester.execute_with_pc(
            &mut branch_harness.executor,
            &mut branch_harness.dense_arena,
            &instructions[5],
            20,
        );
        tester.execute_with_pc(
            &mut branch_harness.executor,
            &mut branch_harness.dense_arena,
            &instructions[6],
            24,
        );
    }
    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.dense_arena,
        &instructions[7],
        28,
    );

    type Record<'a> = (
        &'a mut Rv64BaseAluImmU16AdapterRecord,
        &'a mut AddICoreRecord<BLOCK_FE_WIDTH>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluImmU16AdapterExecutor>::new(),
        );
    type BranchRecord<'a> = (
        &'a mut Rv64BranchAdapterRecord,
        &'a mut BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
    );
    branch_harness
        .dense_arena
        .get_record_seeker::<BranchRecord, _>()
        .transfer_to_matrix_arena(
            &mut branch_harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new(),
        );

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(
        d_transcript.memory_predecessors_host().unwrap(),
        GpuRvrProgram::cpu_memory_predecessors(&execution.transcript).unwrap()
    );
    let (cpu_steps, cpu_ranges) = d_program
        .cpu_replay_plan(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_replay_plan.steps_host().unwrap(), cpu_steps);
    assert_eq!(d_replay_plan.opcode_ranges_host(), &cpu_ranges);
    assert_eq!(
        d_replay_plan
            .opcode_range(BaseAluImmOpcode::ADDI.global_opcode())
            .len(),
        ITERATIONS + 3
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchEqualOpcode::BEQ.global_opcode())
            .len(),
        ITERATIONS + 1
    );
    assert_eq!(
        d_replay_plan
            .opcode_range(BranchEqualOpcode::BNE.global_opcode())
            .len(),
        ITERATIONS
    );
    let mismatched_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    assert!(matches!(
        harness.gpu_chip.generate_proving_ctx_from_rvr(
            &mismatched_program,
            &d_transcript,
            &d_replay_plan,
        ),
        Err(GpuRvrInputError::ProgramMismatch)
    ));
    let (_other_transcript, other_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert!(matches!(
        harness
            .gpu_chip
            .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &other_plan,),
        Err(GpuRvrInputError::SegmentMismatch)
    ));
    let replay_ctx = harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();
    let branch_replay_ctx = branch_harness
        .gpu_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_transcript, &d_replay_plan)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let combined_replay_counts = range_checker.count.to_host_on(device_ctx).unwrap();

    let mut corrupt_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    corrupt_transcript.memory_log[1].value[0] ^= 1;
    let (d_corrupt, d_corrupt_plan) = d_program
        .upload_transcript(&corrupt_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_chip = Rv64AddIChipGpu::new(corrupt_range_checker, tester.timestamp_max_bits());
    corrupt_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt, &d_corrupt_plan)
        .unwrap();
    assert_eq!(d_corrupt.error_code().unwrap(), 9);

    let mut corrupt_branch_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let beq_timestamp = corrupt_branch_transcript.program_log[2].timestamp;
    let beq_read_index = corrupt_branch_transcript
        .memory_log
        .iter()
        .position(|event| event.timestamp == beq_timestamp)
        .unwrap();
    corrupt_branch_transcript.memory_log[beq_read_index].value[0] ^= 1;
    let (d_corrupt_branch, d_corrupt_branch_plan) = d_program
        .upload_transcript(&corrupt_branch_transcript, RvrPreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_branch_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_branch_chip =
        Rv64BranchEqualChipGpu::new(corrupt_branch_range_checker, tester.timestamp_max_bits());
    corrupt_branch_chip
        .generate_proving_ctx_from_rvr(&d_program, &d_corrupt_branch, &d_corrupt_branch_plan)
        .unwrap();
    assert_eq!(d_corrupt_branch.error_code().unwrap(), 28);

    let mut corrupt_branch_predecessor_transcript = RvrPreflightTranscript {
        program_log: execution.transcript.program_log.clone(),
        memory_log: execution.transcript.memory_log.clone(),
        initial_write_log: execution.transcript.initial_write_log.clone(),
    };
    let beq_read_indices = corrupt_branch_predecessor_transcript
        .memory_log
        .iter()
        .enumerate()
        .filter(|(_, event)| {
            event.timestamp == beq_timestamp || event.timestamp == beq_timestamp + 1
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(beq_read_indices.len(), 2);
    for index in beq_read_indices {
        corrupt_branch_predecessor_transcript.memory_log[index].value[0] ^= 1;
    }
    let (d_corrupt_branch_predecessor, d_corrupt_branch_predecessor_plan) = d_program
        .upload_transcript(
            &corrupt_branch_predecessor_transcript,
            RvrPreflightEndpoint::Terminated,
        )
        .unwrap();
    let corrupt_branch_predecessor_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let corrupt_branch_predecessor_chip = Rv64BranchEqualChipGpu::new(
        corrupt_branch_predecessor_range_checker,
        tester.timestamp_max_bits(),
    );
    corrupt_branch_predecessor_chip
        .generate_proving_ctx_from_rvr(
            &d_program,
            &d_corrupt_branch_predecessor,
            &d_corrupt_branch_predecessor_plan,
        )
        .unwrap();
    assert_eq!(d_corrupt_branch_predecessor.error_code().unwrap(), 29);

    let legacy_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_chip =
        Rv64AddIChipGpu::new(legacy_range_checker.clone(), tester.timestamp_max_bits());
    let legacy_ctx = legacy_chip.generate_proving_ctx(harness.dense_arena);
    let legacy_counts = legacy_range_checker.count.to_host_on(device_ctx).unwrap();
    assert_eq!(replay_counts, legacy_counts);
    let raw_count = |count: &F| {
        const { assert!(std::mem::size_of::<F>() == std::mem::size_of::<u32>()) };
        // The CUDA range-check buffer is typed as `F` for shared ownership but
        // kernels update it as an atomic `u32` histogram.
        unsafe { *(std::ptr::from_ref(count).cast::<u32>()) }
    };
    assert_eq!(
        replay_counts.iter().map(raw_count).sum::<u32>(),
        (ITERATIONS as u32 + 3) * 9
    );

    let legacy_branch_range_checker = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        ),
    );
    let legacy_branch_chip = Rv64BranchEqualChipGpu::new(
        legacy_branch_range_checker.clone(),
        tester.timestamp_max_bits(),
    );
    let legacy_branch_ctx = legacy_branch_chip.generate_proving_ctx(branch_harness.dense_arena);
    let legacy_branch_counts = legacy_branch_range_checker
        .count
        .to_host_on(device_ctx)
        .unwrap();
    assert!(combined_replay_counts
        .iter()
        .zip(&replay_counts)
        .zip(&legacy_branch_counts)
        .all(|((combined, addi), branch)| {
            raw_count(combined) - raw_count(addi) == raw_count(branch)
        }));

    let expected_trace =
        <Rv64AddIChip<F> as Chip<MatrixRecordArena<F>, CpuBackend<SC>>>::generate_proving_ctx(
            &harness.cpu_chip,
            harness.matrix_arena,
        )
        .common_main;
    let expected_trace = ColMajorMatrix::from_row_major(&expected_trace);
    device_synchronize().unwrap();
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &legacy_ctx.common_main, device_ctx);
    assert_eq_host_and_device_matrix_col_maj(&expected_trace, &replay_ctx.common_main, device_ctx);

    let expected_branch_trace = <Rv64BranchEqualChip<F> as Chip<
        MatrixRecordArena<F>,
        CpuBackend<SC>,
    >>::generate_proving_ctx(
        &branch_harness.cpu_chip, branch_harness.matrix_arena
    )
    .common_main;
    let replay_branch_trace =
        transport_matrix_d2h_row_major(&branch_replay_ctx.common_main, device_ctx).unwrap();
    let canonical_branch_rows = |matrix: &RowMajorMatrix<F>| {
        // ExecutionState is the first adapter field and timestamp is its second
        // field, so column 1 orders real rows chronologically. Padding rows have
        // timestamp zero and compare equal on both sides.
        let mut rows = (0..matrix.height())
            .map(|row| matrix.row_slice(row).unwrap().to_vec())
            .collect::<Vec<_>>();
        rows.sort_unstable_by_key(|row| row[1].as_canonical_u32());
        rows
    };
    assert_eq!(
        canonical_branch_rows(&expected_branch_trace),
        canonical_branch_rows(&replay_branch_trace),
        "opcode-major replay rows differ from chronological CPU rows"
    );
    let expected_branch_trace = ColMajorMatrix::from_row_major(&expected_branch_trace);
    assert_eq_host_and_device_matrix_col_maj(
        &expected_branch_trace,
        &legacy_branch_ctx.common_main,
        device_ctx,
    );

    tester
        .build()
        .load_air_proving_ctx(Arc::new(harness.air), replay_ctx)
        .load_air_proving_ctx(Arc::new(branch_harness.air), branch_replay_ctx)
        .finalize()
        .simple_test()
        .expect("RVR transcript replay proof failed");
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
#[ignore = "manual release-mode CUDA benchmark"]
fn benchmark_cuda_addi_replay_vs_legacy() {
    const ADDI_ROWS: usize = 1 << 18;
    const WARMUPS: usize = 3;
    const REPETITIONS: usize = 11;
    const LAUNCHES_PER_SAMPLE: usize = 10;

    let reg = |index: usize| index * RV64_REGISTER_NUM_LIMBS;
    let addi = Instruction::<F>::from_usize(
        BaseAluImmOpcode::ADDI.global_opcode(),
        [
            reg(2),
            reg(2),
            0xff_ffff,
            RV64_REGISTER_AS as usize,
            RV64_IMM_AS as usize,
        ],
    );
    let bne = Instruction::<F>::from_isize(
        BranchEqualOpcode::BNE.global_opcode(),
        reg(2) as isize,
        reg(0) as isize,
        -4,
        RV64_REGISTER_AS as isize,
        RV64_REGISTER_AS as isize,
    );
    let terminate =
        Instruction::<F>::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0);
    let program = Program::from_instructions(&[addi.clone(), bne.clone(), terminate]);
    let initial_counter = (ADDI_ROWS as u64).to_le_bytes();
    let init_memory = initial_counter
        .into_iter()
        .enumerate()
        .map(|(offset, byte)| ((RV64_REGISTER_AS, (reg(2) + offset) as u32), byte))
        .collect();
    let exe = VmExe::new(program.clone()).with_init_memory(init_memory);
    let config = Rv64IConfig {
        system: test_system_config(),
        ..Default::default()
    };
    let memory_config = config.system.memory_config.clone();
    let executor = VmExecutor::new(config.clone()).unwrap();
    let rvr = executor.rvr_preflight_instance(&exe, None).unwrap();
    let limits = RvrPreflightLimits::new(2 * ADDI_ROWS + 1, 4 * ADDI_ROWS);
    let execution = rvr.execute(Vec::<Vec<u8>>::new(), limits).unwrap();
    assert_eq!(execution.transcript.program_log.len(), 2 * ADDI_ROWS + 2);
    assert_eq!(execution.transcript.memory_log.len(), 4 * ADDI_ROWS);

    // Build and meter the interpreter once. Keygen, metering, native-code
    // compilation, and initial-state construction are setup costs rather than
    // per-segment preflight work, so none is included in the timed samples.
    let (vm, _) =
        VirtualMachine::new_with_keygen(test_cpu_engine(), crate::Rv64ICpuBuilder, config).unwrap();
    let metered_ctx = vm.build_metered_ctx(&exe);
    let (segments, _) = vm
        .metered_instance(&exe)
        .unwrap()
        .execute_metered(Vec::<Vec<u8>>::new(), metered_ctx)
        .unwrap();
    assert_eq!(segments.len(), 1, "benchmark input must fit one segment");
    let segment = &segments[0];
    assert_eq!(segment.num_insns, (2 * ADDI_ROWS + 1) as u64);
    let mut interpreter = vm.preflight_interpreter(&exe).unwrap();
    let mut rvr_execution_times = Vec::with_capacity(REPETITIONS);
    let mut legacy_preflight_times = Vec::with_capacity(REPETITIONS);
    for sample in 0..WARMUPS + REPETITIONS {
        let time_rvr = || {
            let state = rvr.create_initial_vm_state(Vec::<Vec<u8>>::new());
            let started = std::time::Instant::now();
            let output = rvr.execute_from_state(state, limits).unwrap();
            let elapsed = started.elapsed();
            assert_eq!(output.transcript.program_log.len(), 2 * ADDI_ROWS + 2);
            elapsed
        };
        let mut time_legacy = || {
            let state = vm.create_initial_state(&exe, Vec::<Vec<u8>>::new());
            let started = std::time::Instant::now();
            let output = vm
                .execute_preflight_for(
                    &mut interpreter,
                    state,
                    segment.num_insns,
                    &segment.trace_heights,
                )
                .unwrap();
            let elapsed = started.elapsed();
            assert_eq!(output.to_state.pc(), execution.state.pc());
            elapsed
        };
        let (legacy, rvr) = if sample % 2 == 0 {
            (time_legacy(), time_rvr())
        } else {
            let rvr = time_rvr();
            (time_legacy(), rvr)
        };
        if sample >= WARMUPS {
            legacy_preflight_times.push(legacy);
            rvr_execution_times.push(rvr);
        }
    }

    let mut tester = GpuChipTestBuilder::default();
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        reg(2),
        initial_counter.map(F::from_u8),
    );
    let mut addi_harness = create_cuda_harness_with_capacity(&tester, ADDI_ROWS);
    let mut branch_harness = create_cuda_branch_harness_with_capacity(&tester, ADDI_ROWS);
    for _ in 0..ADDI_ROWS {
        tester.execute_with_pc(
            &mut addi_harness.executor,
            &mut addi_harness.dense_arena,
            &addi,
            0,
        );
        tester.execute_with_pc(
            &mut branch_harness.executor,
            &mut branch_harness.dense_arena,
            &bne,
            4,
        );
    }
    let legacy_addi_record_bytes = addi_harness.dense_arena.allocated().to_vec();
    assert_eq!(
        legacy_addi_record_bytes.len(),
        ADDI_ROWS
            * size_of::<(
                Rv64BaseAluImmU16AdapterRecord,
                AddICoreRecord<BLOCK_FE_WIDTH>,
            )>()
    );
    let legacy_branch_record_bytes = branch_harness.dense_arena.allocated().to_vec();
    assert_eq!(
        legacy_branch_record_bytes.len(),
        ADDI_ROWS
            * size_of::<(
                Rv64BranchAdapterRecord,
                BranchEqualCoreRecord<BLOCK_FE_WIDTH>,
            )>()
    );
    drop(branch_harness);

    let range_checker = tester.range_checker();
    let device_ctx = &range_checker.device_ctx;
    device_ctx.stream.synchronize().unwrap();
    let started = std::time::Instant::now();
    let d_program = GpuRvrProgram::upload(&program, &memory_config, device_ctx).unwrap();
    device_ctx.stream.synchronize().unwrap();
    let static_program_upload = started.elapsed();
    let (gpu_index_requested_peak_live_bytes, gpu_index_requested_steady_live_bytes) = d_program
        .gpu_index_memory_bytes(&execution.transcript)
        .unwrap();

    let mut program_index_times = Vec::with_capacity(REPETITIONS);
    let mut transcript_memory_index_times = Vec::with_capacity(REPETITIONS);
    for sample in 0..WARMUPS + REPETITIONS {
        let (_, _, index_time, upload_time) = d_program
            .upload_transcript_profiled(&execution.transcript, execution.endpoint)
            .unwrap();
        if sample >= WARMUPS {
            program_index_times.push(index_time);
            transcript_memory_index_times.push(upload_time);
        }
    }
    let (d_transcript, d_replay_plan) = d_program
        .upload_transcript(&execution.transcript, execution.endpoint)
        .unwrap();
    assert_eq!(d_transcript.error_code().unwrap(), 0);
    let replay_range = d_replay_plan.opcode_range(BaseAluImmOpcode::ADDI.global_opcode());
    let beq_replay_range = d_replay_plan.opcode_range(BranchEqualOpcode::BEQ.global_opcode());
    let bne_replay_range = d_replay_plan.opcode_range(BranchEqualOpcode::BNE.global_opcode());
    assert_eq!(replay_range.len(), ADDI_ROWS);
    assert!(beq_replay_range.is_empty());
    assert_eq!(bne_replay_range.len(), ADDI_ROWS);

    let mut legacy_upload_times = Vec::with_capacity(REPETITIONS);
    for sample in 0..WARMUPS + REPETITIONS {
        device_ctx.stream.synchronize().unwrap();
        let started = std::time::Instant::now();
        let d_addi_records = legacy_addi_record_bytes.to_device_on(device_ctx).unwrap();
        let d_branch_records = legacy_branch_record_bytes.to_device_on(device_ctx).unwrap();
        device_ctx.stream.synchronize().unwrap();
        if sample >= WARMUPS {
            legacy_upload_times.push(started.elapsed());
        }
        drop(d_addi_records);
        drop(d_branch_records);
    }
    let d_legacy_addi_records = legacy_addi_record_bytes.to_device_on(device_ctx).unwrap();
    let d_legacy_branch_records = legacy_branch_record_bytes.to_device_on(device_ctx).unwrap();

    let addi_trace_width = Rv64BaseAluImmU16AdapterCols::<F>::width()
        + AddICoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let branch_trace_width =
        Rv64BranchAdapterCols::<F>::width() + BranchEqualCoreCols::<F, BLOCK_FE_WIDTH>::width();
    let legacy_addi_trace =
        DeviceMatrix::<F>::with_capacity_on(ADDI_ROWS, addi_trace_width, device_ctx);
    let replay_addi_trace =
        DeviceMatrix::<F>::with_capacity_on(ADDI_ROWS, addi_trace_width, device_ctx);
    let legacy_branch_trace =
        DeviceMatrix::<F>::with_capacity_on(ADDI_ROWS, branch_trace_width, device_ctx);
    let replay_branch_trace =
        DeviceMatrix::<F>::with_capacity_on(ADDI_ROWS, branch_trace_width, device_ctx);
    let legacy_range_checker =
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        );
    let replay_range_checker =
        openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU::new(
            default_var_range_checker_bus(),
            device_ctx.clone(),
        );
    device_ctx.stream.synchronize().unwrap();

    let time_legacy = || {
        benchmark_cuda_launches(device_ctx.stream.as_raw(), LAUNCHES_PER_SAMPLE, || unsafe {
            crate::cuda_abi::addi_cuda::tracegen(
                legacy_addi_trace.buffer(),
                ADDI_ROWS,
                &d_legacy_addi_records,
                &legacy_range_checker.count,
                tester.timestamp_max_bits() as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
            crate::cuda_abi::beq_cuda::tracegen(
                legacy_branch_trace.buffer(),
                ADDI_ROWS,
                &d_legacy_branch_records,
                &legacy_range_checker.count,
                tester.timestamp_max_bits() as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        })
    };
    let time_replay = || {
        benchmark_cuda_launches(device_ctx.stream.as_raw(), LAUNCHES_PER_SAMPLE, || unsafe {
            crate::cuda_abi::addi_cuda::replay_tracegen(
                replay_addi_trace.buffer(),
                ADDI_ROWS,
                d_program.instructions(),
                d_program.pc_base(),
                d_transcript.program_log(),
                d_transcript.memory_log(),
                d_transcript.initial_write_log(),
                d_transcript.memory_predecessors(),
                d_replay_plan.steps(),
                replay_range.start,
                replay_range.len(),
                d_transcript.error_ptr(),
                BaseAluImmOpcode::ADDI.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                RV64_IMM_AS,
                &replay_range_checker.count,
                tester.timestamp_max_bits() as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
            crate::cuda_abi::beq_cuda::replay_tracegen(
                replay_branch_trace.buffer(),
                ADDI_ROWS,
                d_program.instructions(),
                d_program.pc_base(),
                d_transcript.program_log(),
                d_transcript.memory_log(),
                d_transcript.initial_write_log(),
                d_transcript.memory_predecessors(),
                d_replay_plan.steps(),
                beq_replay_range.start,
                beq_replay_range.len(),
                bne_replay_range.start,
                bne_replay_range.len(),
                d_transcript.error_ptr(),
                BranchEqualOpcode::BEQ.global_opcode().as_usize() as u32,
                BranchEqualOpcode::BNE.global_opcode().as_usize() as u32,
                RV64_REGISTER_AS,
                &replay_range_checker.count,
                tester.timestamp_max_bits() as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        })
    };

    let mut legacy_kernel_ms = Vec::with_capacity(REPETITIONS);
    let mut replay_kernel_ms = Vec::with_capacity(REPETITIONS);
    for sample in 0..WARMUPS + REPETITIONS {
        let (legacy, replay) = if sample % 2 == 0 {
            (time_legacy(), time_replay())
        } else {
            let replay = time_replay();
            (time_legacy(), replay)
        };
        if sample >= WARMUPS {
            legacy_kernel_ms.push(legacy);
            replay_kernel_ms.push(replay);
        }
    }
    assert_eq!(d_transcript.error_code().unwrap(), 0);

    let (program_index_p10, program_index_median, program_index_p90) =
        duration_percentiles(program_index_times);
    let (rvr_execution_p10, rvr_execution_median, rvr_execution_p90) =
        duration_percentiles(rvr_execution_times);
    let (legacy_preflight_p10, legacy_preflight_median, legacy_preflight_p90) =
        duration_percentiles(legacy_preflight_times);
    let (transcript_memory_index_p10, transcript_memory_index_median, transcript_memory_index_p90) =
        duration_percentiles(transcript_memory_index_times);
    let (legacy_upload_p10, legacy_upload_median, legacy_upload_p90) =
        duration_percentiles(legacy_upload_times);
    let (legacy_kernel_p10, legacy_kernel_median, legacy_kernel_p90) =
        f64_percentiles(legacy_kernel_ms);
    let (replay_kernel_p10, replay_kernel_median, replay_kernel_p90) =
        f64_percentiles(replay_kernel_ms);
    let micros = |duration: Duration| duration.as_secs_f64() * 1_000_000.0;
    let gpu_replay_total_us = micros(program_index_median)
        + micros(transcript_memory_index_median)
        + replay_kernel_median * 1000.0;
    let gpu_legacy_total_us = micros(legacy_upload_median) + legacy_kernel_median * 1000.0;
    let summed_replay_slice_total_us = micros(rvr_execution_median) + gpu_replay_total_us;
    let summed_legacy_slice_total_us = micros(legacy_preflight_median) + gpu_legacy_total_us;
    let transcript_bytes = std::mem::size_of_val(execution.transcript.program_log.as_slice())
        + std::mem::size_of_val(execution.transcript.memory_log.as_slice())
        + std::mem::size_of_val(execution.transcript.initial_write_log.as_slice());
    let derived_bytes = execution.transcript.memory_log.len() * size_of::<u32>()
        + (execution.transcript.program_log.len() - 1) * 2 * size_of::<u32>();
    let legacy_record_bytes = legacy_addi_record_bytes.len() + legacy_branch_record_bytes.len();
    let trace_bytes = ADDI_ROWS * (addi_trace_width + branch_trace_width) * size_of::<F>();
    println!(
        "RVR_RISCV_SLICE_GPU_BENCH addi_rows={ADDI_ROWS} branch_rows={ADDI_ROWS} guest_insns={} warmups={WARMUPS} repetitions={REPETITIONS} launches_per_sample={LAUNCHES_PER_SAMPLE} rvr_execution_p10_us={:.3} rvr_execution_median_us={:.3} rvr_execution_p90_us={:.3} legacy_preflight_p10_us={:.3} legacy_preflight_median_us={:.3} legacy_preflight_p90_us={:.3} static_program_upload_us={:.3} program_index_p10_us={:.3} program_index_median_us={:.3} program_index_p90_us={:.3} transcript_memory_index_p10_us={:.3} transcript_memory_index_median_us={:.3} transcript_memory_index_p90_us={:.3} legacy_record_h2d_p10_us={:.3} legacy_record_h2d_median_us={:.3} legacy_record_h2d_p90_us={:.3} replay_kernel_p10_us={:.3} replay_kernel_median_us={:.3} replay_kernel_p90_us={:.3} legacy_kernel_p10_us={:.3} legacy_kernel_median_us={:.3} legacy_kernel_p90_us={:.3} replay_over_legacy_kernel={:.3} gpu_replay_total_us={gpu_replay_total_us:.3} gpu_legacy_total_us={gpu_legacy_total_us:.3} summed_replay_slice_total_us={summed_replay_slice_total_us:.3} summed_legacy_slice_total_us={summed_legacy_slice_total_us:.3} modeled_slice_speedup={:.3} transcript_bytes={transcript_bytes} derived_bytes={derived_bytes} gpu_index_requested_peak_live_bytes={gpu_index_requested_peak_live_bytes} gpu_index_requested_steady_live_bytes={gpu_index_requested_steady_live_bytes} legacy_record_bytes={legacy_record_bytes} trace_bytes={trace_bytes}",
        2 * ADDI_ROWS + 1,
        micros(rvr_execution_p10),
        micros(rvr_execution_median),
        micros(rvr_execution_p90),
        micros(legacy_preflight_p10),
        micros(legacy_preflight_median),
        micros(legacy_preflight_p90),
        micros(static_program_upload),
        micros(program_index_p10),
        micros(program_index_median),
        micros(program_index_p90),
        micros(transcript_memory_index_p10),
        micros(transcript_memory_index_median),
        micros(transcript_memory_index_p90),
        micros(legacy_upload_p10),
        micros(legacy_upload_median),
        micros(legacy_upload_p90),
        replay_kernel_p10 * 1000.0,
        replay_kernel_median * 1000.0,
        replay_kernel_p90 * 1000.0,
        legacy_kernel_p10 * 1000.0,
        legacy_kernel_median * 1000.0,
        legacy_kernel_p90 * 1000.0,
        replay_kernel_median / legacy_kernel_median,
        summed_legacy_slice_total_us / summed_replay_slice_total_us,
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rv64_addi_rs1_memory_binding_negative_test() {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some([0; RV64_REGISTER_NUM_LIMBS]),
        Some([0; RV64_REGISTER_NUM_LIMBS]),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut AddICoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        // These limbs represent 2^64, so ADDI 0 still produces zero and all core
        // constraints hold. Only the rs1 memory-read interaction rejects the row.
        cols.rs1 = NONCANONICAL_ZERO.map(F::from_u32);
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
