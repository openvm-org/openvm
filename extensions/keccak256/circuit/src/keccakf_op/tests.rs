use std::{
    array::from_fn,
    mem::size_of,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use openvm_circuit::{
    arch::{
        testing::{
            memory::{gen_pointer, gen_register_pointer},
            TestBuilder, TestChipHarness, TestPreflight, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
        },
        ExecutionBridge, Executor, MemoryConfig, Postflight, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::get_random_message,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::KeccakfOpcode;
use openvm_stark_backend::{
    interaction::{BusIndex, PermutationCheckBus},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::Matrix,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use rand::Rng;
use tiny_keccak::keccakf;
#[cfg(all(feature = "cuda", feature = "rvr"))]
use {
    openvm_circuit::arch::{
        cuda::postflight::GpuPostflightProgram, PreflightHistory, PreflightMemoryLog,
    },
    openvm_instructions::{program::Program, SystemOpcode},
    rvr_state::{
        PreflightInitialWrite, PreflightMemoryEvent, PreflightProgramEvent, PREFLIGHT_WRITE_BIT,
    },
};

use crate::{
    keccakf_op::{generate_trace_from_postflight, KeccakfExecutor, KeccakfOpAir, KeccakfOpChip},
    keccakf_perm::{
        generate_trace_from_postflight as generate_perm_trace_from_postflight, KeccakfPermAir,
        KeccakfPermChip,
    },
    KECCAK_WIDTH_BYTES, KECCAK_WIDTH_U64S,
};

type F = BabyBear;
/// Harness without KeccakfPeriphery*
type Harness = TestChipHarness<F, KeccakfExecutor, KeccakfOpAir, KeccakfOpChip<F>>;
const MAX_TRACE_ROWS: usize = 4096;
const KECCAKF_STATE_BUS: BusIndex = 13;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (KeccakfOpAir, KeccakfExecutor, KeccakfOpChip<F>) {
    let executor = KeccakfExecutor::new(KeccakfOpcode::CLASS_OFFSET, address_bits);
    let empty_records = Arc::new(Mutex::new(Vec::new()));
    let op_air = KeccakfOpAir::new(
        execution_bridge,
        memory_bridge,
        PermutationCheckBus::new(KECCAKF_STATE_BUS),
        range_checker_chip.bus(),
        address_bits,
        KeccakfOpcode::CLASS_OFFSET,
    );
    let op_chip = KeccakfOpChip::new(
        range_checker_chip,
        address_bits,
        memory_helper,
        empty_records,
    );
    (op_air, executor, op_chip)
}

struct TestHarness {
    harness: Harness,
    bitwise: (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
    perm: (KeccakfPermAir, KeccakfPermChip),
}

fn create_test_harness(tester: &mut VmChipTestBuilder<F>) -> TestHarness {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (op_air, executor, op_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let shared_preimages = op_chip.shared_preimages.clone();

    let harness = Harness::with_capacity(
        executor,
        op_air,
        op_chip,
        MAX_TRACE_ROWS,
        |chip, postflight| {
            let mut previous = std::mem::take(&mut *chip.shared_preimages.lock().unwrap());
            let trace = generate_trace_from_postflight(chip, postflight)?;
            previous.extend(std::mem::take(&mut *chip.shared_preimages.lock().unwrap()));
            *chip.shared_preimages.lock().unwrap() = previous;
            Ok(trace)
        },
    );

    let perm_air = KeccakfPermAir::new(op_air.keccakf_state_bus);
    let perm_chip = KeccakfPermChip::new(shared_preimages);

    TestHarness {
        harness,
        bitwise: (bitwise_chip.air, bitwise_chip),
        perm: (perm_air, perm_chip),
    }
}

fn set_and_execute_single_perm<E: Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut TestPreflight<F>,
    rng: &mut StdRng,
    opcode: KeccakfOpcode,
) {
    const MAX_LEN: usize = KECCAK_WIDTH_BYTES;
    const U64_NUM_BYTES: usize = size_of::<u64>();
    let rand_buffer = get_random_message(rng, MAX_LEN);
    let mut rand_buffer_arr = [0u8; MAX_LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    let rd = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let buffer_ptr = gen_pointer(rng, MAX_LEN);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rd,
        (buffer_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    let rand_buffer_arr_f = rand_buffer_arr.map(F::from_u8);

    for i in 0..(MAX_LEN / MEMORY_BLOCK_BYTES) {
        let buffer_chunk: [F; MEMORY_BLOCK_BYTES] = rand_buffer_arr_f
            [MEMORY_BLOCK_BYTES * i..MEMORY_BLOCK_BYTES * (i + 1)]
            .try_into()
            .expect("slice has correct length");
        tester.write_bytes(
            RV64_MEMORY_AS as usize,
            buffer_ptr + MEMORY_BLOCK_BYTES * i,
            buffer_chunk,
        );
    }

    tester.execute(
        executor,
        preflight,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [rd, 0, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    let mut output_buffer = [0u8; MAX_LEN];

    for i in 0..(MAX_LEN / MEMORY_BLOCK_BYTES) {
        let output_chunk: [F; MEMORY_BLOCK_BYTES] =
            tester.read_bytes(RV64_MEMORY_AS as usize, buffer_ptr + MEMORY_BLOCK_BYTES * i);
        let output_chunk = output_chunk.map(|x| x.as_canonical_u32() as u8);
        output_buffer[MEMORY_BLOCK_BYTES * i..MEMORY_BLOCK_BYTES * (i + 1)]
            .copy_from_slice(&output_chunk);
    }
    let mut state: [u64; KECCAK_WIDTH_U64S] = from_fn(|i| {
        u64::from_le_bytes(
            rand_buffer[U64_NUM_BYTES * i..U64_NUM_BYTES * (i + 1)]
                .try_into()
                .unwrap(),
        )
    });
    keccakf(&mut state);
    let expected_out = state.iter().flat_map(|w| w.to_le_bytes()).collect_vec();
    assert_eq!(&output_buffer[..], &expected_out[..]);
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_keccakf_positive_tests() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        perm,
    } = create_test_harness(&mut tester);

    let num_ops: usize = 100;
    for _ in 0..num_ops {
        set_and_execute_single_perm(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            KeccakfOpcode::KECCAKF,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(perm)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

fn keccakf_postflight_fixture() -> TestHarness {
    let mut tester = VmChipTestBuilder::default();
    let mut test_harness = create_test_harness(&mut tester);
    let instruction = Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [8, 0, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
    );
    let sentinel = instruction.clone();
    let block = |bytes: [u8; MEMORY_BLOCK_BYTES]| {
        std::array::from_fn(|index| u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]))
    };
    unsafe {
        let memory = &mut tester.memory.memory.data;
        memory.write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 4, block((0x100u64).to_le_bytes()));
        for word_index in 0..KECCAK_WIDTH_BYTES / MEMORY_BLOCK_BYTES {
            memory.write::<u16, BLOCK_FE_WIDTH>(
                RV64_MEMORY_AS,
                0x80 + (word_index * BLOCK_FE_WIDTH) as u32,
                block(std::array::from_fn(|byte| {
                    (word_index * MEMORY_BLOCK_BYTES + byte) as u8
                })),
            );
        }
    }
    tester.execute_with_pc(
        &mut test_harness.harness.executor,
        &mut test_harness.harness.preflight,
        &instruction,
        0,
    );
    let _ = sentinel;
    test_harness
}

#[test]
fn postflight_keccakf_trace_generation_succeeds() {
    let TestHarness {
        harness,
        perm,
        bitwise: _,
    } = keccakf_postflight_fixture();
    let execution = &harness.preflight.executions[0];
    let memory_config = MemoryConfig::default();
    let postflight =
        Postflight::new_for_test(&execution.program, &execution.history, &memory_config).unwrap();

    let actual_op = generate_trace_from_postflight(&harness.chip, &postflight).unwrap();
    let actual_perm = generate_perm_trace_from_postflight(&perm.1, &postflight).unwrap();

    assert_eq!(actual_op.height(), 1);
    assert!(!actual_perm.values.is_empty());
}

#[test]
fn postflight_keccakf_rejects_corrupt_write() {
    let test_harness = keccakf_postflight_fixture();
    let execution = &test_harness.harness.preflight.executions[0];
    let mut history = execution.history.clone();
    history.memory.accesses.last_mut().unwrap().value[0] ^= 1;
    let memory_config = MemoryConfig::default();
    let postflight =
        Postflight::new_for_test(&execution.program, &history, &memory_config).unwrap();
    let error =
        generate_trace_from_postflight(&test_harness.harness.chip, &postflight).unwrap_err();

    assert!(error.to_string().contains("unexpected write"));
}

// ////////////////////////////////////////////////////////////////////////////////////
// CUDA TESTS
// ////////////////////////////////////////////////////////////////////////////////////
#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_circuit::arch::testing::{
    default_bitwise_lookup_bus, GpuChipTestBuilder, GpuChipTester, GpuTestChipHarness,
};

#[cfg(all(feature = "cuda", feature = "rvr"))]
use crate::cuda::{
    KeccakfOpChipGpu, KeccakfPermChipGpu, SharedKeccakfState, SharedKeccakfStateGpu,
};

#[cfg(all(feature = "cuda", feature = "rvr"))]
type GpuHarness =
    GpuTestChipHarness<F, KeccakfExecutor, KeccakfOpAir, KeccakfOpChipGpu, KeccakfOpChip<F>>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
type PermGpuHarness =
    GpuTestChipHarness<F, (), KeccakfPermAir, KeccakfPermChipGpu, KeccakfPermChip>;

#[cfg(all(feature = "cuda", feature = "rvr"))]
struct CudaTestHarness {
    op_harness: GpuHarness,
    perm_harness: PermGpuHarness,
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
impl CudaTestHarness {
    fn into_parts(mut self) -> (GpuHarness, PermGpuHarness) {
        self.perm_harness.preflight = self.op_harness.preflight.clone();
        (self.op_harness, self.perm_harness)
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> CudaTestHarness {
    let dummy_range_checker_chip = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChip::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
        ),
    );

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let keccakf_state_bus = air.keccakf_state_bus;
    let cpu_shared_preimages = cpu_chip.shared_preimages.clone();

    let shared_state: SharedKeccakfStateGpu = Arc::new(Mutex::new(SharedKeccakfState::default()));

    let gpu_chip = KeccakfOpChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
        shared_state.clone(),
    );

    let op_harness =
        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_TRACE_ROWS)
            .with_trace_generators(
                generate_trace_from_postflight,
                |chip, program, transcript, plan| {
                    chip.generate_proving_ctx_from_postflight(program, transcript, plan)
                },
            );

    let perm_air = KeccakfPermAir::new(keccakf_state_bus);
    let device_ctx = tester.range_checker().device_ctx.clone();
    let perm_gpu_chip = KeccakfPermChipGpu::new(shared_state, device_ctx);
    let perm_cpu_chip = KeccakfPermChip::new(cpu_shared_preimages);
    let perm_harness = GpuTestChipHarness::with_capacity(
        (),
        perm_air,
        perm_gpu_chip,
        perm_cpu_chip,
        MAX_TRACE_ROWS,
    )
    .with_trace_generators(
        generate_perm_trace_from_postflight,
        |chip, program, transcript, plan| {
            chip.generate_proving_ctx_from_postflight(program, transcript, plan)
        },
    )
    .without_memory_balance();

    CudaTestHarness {
        op_harness,
        perm_harness,
    }
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn load_cuda_harness(tester: GpuChipTester, harness: CudaTestHarness) -> GpuChipTester {
    let (op_harness, perm_harness) = harness.into_parts();
    tester
        .load_gpu_harness(op_harness)
        .load_gpu_harness(perm_harness)
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
fn cuda_set_and_execute(
    tester: &mut GpuChipTestBuilder,
    executor: &mut KeccakfExecutor,
    preflight: &mut TestPreflight<F>,
    rng: &mut StdRng,
) {
    const KECCAK_STATE_BYTES: usize = 200;

    let buffer_reg = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let buffer_ptr = gen_pointer(rng, KECCAK_STATE_BYTES);

    tester.write_bytes(
        1,
        buffer_reg,
        (buffer_ptr as u64).to_le_bytes().map(F::from_u8),
    );

    let state_data: Vec<u8> = (0..KECCAK_STATE_BYTES).map(|_| rng.random()).collect();
    for (i, chunk) in state_data.chunks(MEMORY_BLOCK_BYTES).enumerate() {
        let mut word = [F::ZERO; MEMORY_BLOCK_BYTES];
        for (j, &byte) in chunk.iter().enumerate() {
            word[j] = F::from_u8(byte);
        }
        tester.write_bytes(2, buffer_ptr + i * MEMORY_BLOCK_BYTES, word);
    }

    let instruction = Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [buffer_reg, 0, 0, 1, 2],
    );

    tester.execute(executor, preflight, &instruction);
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_keccakf_cuda_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let num_ops: usize = 3;
    let mut harnesses = Vec::with_capacity(num_ops);
    for _ in 0..num_ops {
        let mut harness = create_cuda_harness(&tester);
        cuda_set_and_execute(
            &mut tester,
            &mut harness.op_harness.executor,
            &mut harness.op_harness.preflight,
            &mut rng,
        );
        harnesses.push(harness);
    }

    let mut tester = tester.build();
    for harness in harnesses {
        tester = load_cuda_harness(tester, harness);
    }
    tester.finalize().simple_test().unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_keccakf_cuda_tracegen_single() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    cuda_set_and_execute(
        &mut tester,
        &mut harness.op_harness.executor,
        &mut harness.op_harness.preflight,
        &mut rng,
    );

    load_cuda_harness(tester.build(), harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_keccakf_cuda_tracegen_zero_state() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    const KECCAK_STATE_BYTES: usize = 200;

    let buffer_reg = gen_register_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    let buffer_ptr = gen_pointer(&mut rng, KECCAK_STATE_BYTES);

    tester.write_bytes(
        1,
        buffer_reg,
        (buffer_ptr as u64).to_le_bytes().map(F::from_u8),
    );

    for i in 0..(KECCAK_STATE_BYTES / MEMORY_BLOCK_BYTES) {
        tester.write_bytes(
            2,
            buffer_ptr + i * MEMORY_BLOCK_BYTES,
            [F::ZERO; MEMORY_BLOCK_BYTES],
        );
    }

    let instruction = Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [buffer_reg, 0, 0, 1, 2],
    );

    tester.execute(
        &mut harness.op_harness.executor,
        &mut harness.op_harness.preflight,
        &instruction,
    );

    load_cuda_harness(tester.build(), harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_keccakf_preflight_replay_accepts_valid_transcript_and_rejects_corruption() {
    let buffer_reg = 8usize;
    let buffer_ptr = 0x100u32;
    let keccakf_instruction = Instruction::<F>::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [
            buffer_reg,
            0,
            0,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    let instructions = [
        keccakf_instruction,
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let memory_config = openvm_circuit::arch::MemoryConfig::default();
    let block = |bytes: &[u8]| {
        std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
    };
    let initial_bytes = (0..KECCAK_WIDTH_BYTES)
        .map(|offset| (offset as u8).wrapping_mul(29))
        .collect::<Vec<_>>();
    let mut state = std::array::from_fn::<_, KECCAK_WIDTH_U64S, _>(|i| {
        u64::from_le_bytes(initial_bytes[i * 8..][..8].try_into().unwrap())
    });
    keccakf(&mut state);
    let postimage = state
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let mut memory_log = vec![PreflightMemoryEvent {
        timestamp: 1,
        address_space_and_kind: RV64_REGISTER_AS,
        pointer: buffer_reg as u32 / 2,
        value: block(&(buffer_ptr as u64).to_le_bytes()),
    }];
    let mut initial_write_log = Vec::with_capacity(KECCAK_WIDTH_BYTES / 8);
    for i in 0..KECCAK_WIDTH_BYTES / 8 {
        let pointer = buffer_ptr / 2 + (i * 4) as u32;
        let initial_value = block(&initial_bytes[i * 8..][..8]);
        initial_write_log.push(PreflightInitialWrite {
            address_space: RV64_MEMORY_AS,
            pointer,
            initial_value,
        });
        memory_log.push(PreflightMemoryEvent {
            timestamp: 2 + i as u32,
            address_space_and_kind: RV64_MEMORY_AS | PREFLIGHT_WRITE_BIT,
            pointer,
            value: block(&postimage[i * 8..][..8]),
        });
    }
    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 27,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 27,
            },
        ],
        memory: PreflightMemoryLog {
            accesses: memory_log,
            initial_writes: initial_write_log,
            ..Default::default()
        },
    };

    let tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let shared_state = Arc::new(Mutex::new(SharedKeccakfState::default()));
    let op_chip = KeccakfOpChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
        shared_state,
    );

    let device_ctx = &tester.range_checker().device_ctx;
    let gpu_program = GpuPostflightProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_history_for_test(&program, &history, Some(0))
        .unwrap();
    let _op_ctx = op_chip
        .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
        .unwrap();
    assert_eq!(gpu_transcript.error_code().unwrap(), 0);

    let mut corrupt = history;
    corrupt.memory.accesses[1].value[0] ^= 1;
    let (gpu_corrupt, corrupt_plan) = gpu_program
        .upload_history_for_test(&program, &corrupt, Some(0))
        .unwrap();
    let corrupt_shared = Arc::new(Mutex::new(SharedKeccakfState::default()));
    let corrupt_chip = KeccakfOpChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
        corrupt_shared,
    );
    corrupt_chip
        .generate_proving_ctx_from_postflight(&gpu_program, &gpu_corrupt, &corrupt_plan)
        .unwrap();
    assert_eq!(gpu_corrupt.error_code().unwrap(), 811);
}
