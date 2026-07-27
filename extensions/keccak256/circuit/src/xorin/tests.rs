use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
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
    program::{Program, DEFAULT_PC_STEP},
    riscv::{RV64_BYTE_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_stark_backend::{
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
    openvm_circuit::arch::rvr::{cuda::GpuPostflightProgram, PreflightEndpoint, PreflightEventLog},
    openvm_instructions::SystemOpcode,
    rvr_state::{PreflightMemoryEvent, PREFLIGHT_WRITE_BIT},
};

use crate::{
    xorin::{
        air::XorinVmAir,
        columns::{XorinVmCols, NUM_XORIN_VM_COLS},
        trace::generate_trace_from_postflight,
        XorinVmChip, XorinVmExecutor, XorinVmFiller,
    },
    KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS,
};

type F = BabyBear;
type Harness = TestChipHarness<F, XorinVmExecutor, XorinVmAir, XorinVmChip<F>>;
const MAX_TRACE_ROWS: usize = 4096;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (XorinVmAir, XorinVmExecutor, XorinVmChip<F>) {
    let air = XorinVmAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        range_checker_chip.bus(),
        address_bits,
        XorinOpcode::CLASS_OFFSET,
    );

    let executor = XorinVmExecutor::new(XorinOpcode::CLASS_OFFSET, address_bits);
    let chip = XorinVmChip::new(
        XorinVmFiller::new(bitwise_chip, range_checker_chip, address_bits),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_harness(
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
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );

    let harness = Harness::with_capacity(
        executor,
        air,
        chip,
        MAX_TRACE_ROWS,
        generate_trace_from_postflight,
    );

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<E: Executor<F> + Clone>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    preflight: &mut TestPreflight<F>,
    rng: &mut StdRng,
    opcode: XorinOpcode,
    buffer_length: Option<usize>,
) {
    const MAX_LEN: usize = KECCAK_RATE_BYTES;

    let buffer_length = match buffer_length {
        Some(length) => length,
        None => MAX_LEN,
    };

    assert!(buffer_length.is_multiple_of(MEMORY_BLOCK_BYTES));

    let rand_buffer = get_random_message(rng, MAX_LEN);
    let mut rand_buffer_arr = [0u8; MAX_LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    let rand_input = get_random_message(rng, MAX_LEN);
    let mut rand_input_arr = [0u8; MAX_LEN];
    rand_input_arr.copy_from_slice(&rand_input);

    use openvm_circuit::arch::testing::memory::{gen_distinct_register_pointers, gen_pointer};
    let [rd, rs1, rs2] = gen_distinct_register_pointers(rng, RV64_REGISTER_NUM_LIMBS);

    // Align buffer/input pointers to MEMORY_BLOCK_BYTES-byte blocks for memory bus compatibility
    let num_blocks = buffer_length.div_ceil(MEMORY_BLOCK_BYTES);
    let aligned_len = num_blocks * MEMORY_BLOCK_BYTES;
    let buffer_ptr = gen_pointer(rng, aligned_len);
    let input_ptr = gen_pointer(rng, aligned_len);

    let rand_buffer_arr_f = rand_buffer_arr.map(F::from_u8);
    let rand_input_arr_f = rand_input_arr.map(F::from_u8);

    // Write memory in MEMORY_BLOCK_BYTES-byte blocks; for the last partial block, pad with zeros
    for i in 0..num_blocks {
        let start = MEMORY_BLOCK_BYTES * i;
        let end = std::cmp::min(start + MEMORY_BLOCK_BYTES, MAX_LEN);
        let mut buffer_chunk = [F::ZERO; MEMORY_BLOCK_BYTES];
        for (j, &v) in rand_buffer_arr_f[start..end].iter().enumerate() {
            buffer_chunk[j] = v;
        }
        tester.write_bytes(
            RV64_MEMORY_AS as usize,
            buffer_ptr + MEMORY_BLOCK_BYTES * i,
            buffer_chunk,
        );

        let mut input_chunk = [F::ZERO; MEMORY_BLOCK_BYTES];
        for (j, &v) in rand_input_arr_f[start..end].iter().enumerate() {
            input_chunk[j] = v;
        }
        tester.write_bytes(
            RV64_MEMORY_AS as usize,
            input_ptr + MEMORY_BLOCK_BYTES * i,
            input_chunk,
        );
    }

    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rd,
        (buffer_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rs1,
        (input_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        rs2,
        (buffer_length as u64).to_le_bytes().map(F::from_u8),
    );
    tester.execute(
        executor,
        preflight,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                rd,
                rs1,
                rs2,
                RV64_REGISTER_AS as usize,
                RV64_MEMORY_AS as usize,
            ],
        ),
    );

    let mut expected_output = [0u8; MAX_LEN];
    for i in 0..buffer_length {
        expected_output[i] = rand_buffer_arr[i] ^ rand_input_arr[i];
    }

    let mut output_buffer = [F::from_u8(0); MAX_LEN];

    for i in 0..num_blocks {
        let output_chunk: [F; MEMORY_BLOCK_BYTES] =
            tester.read_bytes(RV64_MEMORY_AS as usize, buffer_ptr + MEMORY_BLOCK_BYTES * i);
        let start = MEMORY_BLOCK_BYTES * i;
        let end = std::cmp::min(start + MEMORY_BLOCK_BYTES, MAX_LEN);
        output_buffer[start..end].copy_from_slice(&output_chunk[..end - start]);
    }

    for i in 0..buffer_length {
        assert_eq!(F::from_u8(expected_output[i]), output_buffer[i]);
    }
}

#[test]
fn xorin_chip_positive_tests() {
    let num_ops: usize = 100;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    for _ in 0..num_ops {
        let buffer_length = Some(rng.random_range(1..=KECCAK_RATE_MEM_OPS) * MEMORY_BLOCK_BYTES);

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.preflight,
            &mut rng,
            XorinOpcode::XORIN,
            buffer_length,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

fn run_xorin_chip_negative_test(prank: impl Fn(&mut XorinVmCols<F>)) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    let buffer_length = Some(rng.random_range(1..=KECCAK_RATE_MEM_OPS) * MEMORY_BLOCK_BYTES);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.preflight,
        &mut rng,
        XorinOpcode::XORIN,
        buffer_length,
    );

    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut XorinVmCols<F> = values.split_at_mut(NUM_XORIN_VM_COLS).0.borrow_mut();
        prank(cols);
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
fn xorin_wrong_output_negative_test() {
    run_xorin_chip_negative_test(|cols| {
        cols.sponge.postimage_buffer_bytes[0] += F::ONE;
    });
}

#[test]
fn xorin_wrong_len_limb_negative_test() {
    run_xorin_chip_negative_test(|cols| {
        cols.instruction.len_limb += F::ONE;
    });
}

fn xorin_postflight_fixture() -> Harness {
    const LEN: usize = 16;

    let mut tester = VmChipTestBuilder::default();
    let (mut harness, _) = create_test_harness(&mut tester);
    let instruction = Instruction::from_usize(
        XorinOpcode::XORIN.global_opcode(),
        [
            8,
            16,
            24,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    let block = |bytes: [u8; MEMORY_BLOCK_BYTES]| {
        std::array::from_fn(|index| u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]))
    };
    unsafe {
        let memory = &mut tester.memory.memory.data;
        memory.write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 4, block((0x100u64).to_le_bytes()));
        memory.write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 8, block((0x200u64).to_le_bytes()));
        memory.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            12,
            block((LEN as u64).to_le_bytes()),
        );
        for index in 0..LEN / MEMORY_BLOCK_BYTES {
            memory.write::<u16, BLOCK_FE_WIDTH>(
                RV64_MEMORY_AS,
                0x80 + (index * BLOCK_FE_WIDTH) as u32,
                block(std::array::from_fn(|byte| {
                    (index * MEMORY_BLOCK_BYTES + byte) as u8
                })),
            );
            memory.write::<u16, BLOCK_FE_WIDTH>(
                RV64_MEMORY_AS,
                0x100 + (index * BLOCK_FE_WIDTH) as u32,
                block(std::array::from_fn(|byte| {
                    (0xa0 + index * MEMORY_BLOCK_BYTES + byte) as u8
                })),
            );
        }
    }
    tester.execute_with_pc(
        &mut harness.executor,
        &mut harness.preflight,
        &instruction,
        0,
    );
    harness
}

#[test]
fn postflight_xorin_trace_generation_succeeds() {
    let harness = xorin_postflight_fixture();
    let execution = &harness.preflight.executions[0];
    let memory_config = MemoryConfig::default();
    let postflight =
        Postflight::new_for_test(&execution.program, &execution.history, &memory_config).unwrap();
    let actual = generate_trace_from_postflight(&harness.chip, &postflight).unwrap();

    assert_eq!(actual.width(), NUM_XORIN_VM_COLS);
    assert_eq!(actual.height(), 1);
}

#[test]
fn postflight_xorin_rejects_corrupt_write() {
    let harness = xorin_postflight_fixture();
    let execution = &harness.preflight.executions[0];
    let mut history = execution.history.clone();
    history.memory.accesses.last_mut().unwrap().value[0] ^= 1;
    let memory_config = MemoryConfig::default();
    let postflight =
        Postflight::new_for_test(&execution.program, &history, &memory_config).unwrap();
    let error = generate_trace_from_postflight(&harness.chip, &postflight).unwrap_err();

    assert!(error.to_string().contains("unexpected write"));
}

// ////////////////////////////////////////////////////////////////////////////////////
// CUDA TESTS
// ////////////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "cuda")]
use openvm_circuit::arch::{
    testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
    DenseRecordArena,
};

#[cfg(feature = "cuda")]
use crate::{cuda::XorinVmChipGpu, xorin::trace::XorinVmRecordMut};

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, XorinVmExecutor, XorinVmAir, XorinVmChipGpu, XorinVmChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_checker_chip = Arc::new(
        openvm_circuit_primitives::var_range::VariableRangeCheckerChip::new(
            openvm_circuit::arch::testing::default_var_range_checker_bus(),
        ),
    );

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        dummy_bitwise_chip,
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );

    let gpu_chip = XorinVmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_TRACE_ROWS)
}

#[cfg(feature = "cuda")]
fn cuda_set_and_execute(
    tester: &mut GpuChipTestBuilder,
    executor: &mut XorinVmExecutor,
    arena: &mut DenseRecordArena,
    rng: &mut StdRng,
    len: Option<usize>,
) {
    use openvm_circuit::arch::testing::memory::{gen_distinct_register_pointers, gen_pointer};

    let len = len.unwrap_or_else(|| rng.random_range(1..=KECCAK_RATE_MEM_OPS) * MEMORY_BLOCK_BYTES);
    if len == 0 {
        return;
    }

    let [buffer_reg, input_reg, len_reg] =
        gen_distinct_register_pointers(rng, RV64_REGISTER_NUM_LIMBS);

    let buffer_ptr = gen_pointer(rng, len);
    let input_ptr = gen_pointer(rng, len);

    tester.write_bytes(
        1,
        buffer_reg,
        (buffer_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(
        1,
        input_reg,
        (input_ptr as u64).to_le_bytes().map(F::from_u8),
    );
    tester.write_bytes(1, len_reg, (len as u64).to_le_bytes().map(F::from_u8));

    let buffer_data: Vec<u8> = (0..len).map(|_| rng.random()).collect();
    for (i, chunk) in buffer_data.chunks(MEMORY_BLOCK_BYTES).enumerate() {
        let mut word = [F::ZERO; MEMORY_BLOCK_BYTES];
        for (j, &byte) in chunk.iter().enumerate() {
            word[j] = F::from_u8(byte);
        }
        tester.write_bytes(2, buffer_ptr + i * MEMORY_BLOCK_BYTES, word);
    }

    let input_data: Vec<u8> = (0..len).map(|_| rng.random()).collect();
    for (i, chunk) in input_data.chunks(MEMORY_BLOCK_BYTES).enumerate() {
        let mut word = [F::ZERO; MEMORY_BLOCK_BYTES];
        for (j, &byte) in chunk.iter().enumerate() {
            word[j] = F::from_u8(byte);
        }
        tester.write_bytes(2, input_ptr + i * MEMORY_BLOCK_BYTES, word);
    }

    let instruction = Instruction::from_usize(
        XorinOpcode::XORIN.global_opcode(),
        [buffer_reg, input_reg, len_reg, 1, 2],
    );

    tester.execute(executor, arena, &instruction);
}

#[cfg(feature = "cuda")]
#[test]
fn test_xorin_cuda_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    let num_ops: usize = 5;
    for _ in 0..num_ops {
        cuda_set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            None,
        );
    }

    for len in [8, 16, 24, 32, 64, 128, 136] {
        cuda_set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            Some(len),
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<XorinVmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn test_xorin_cuda_tracegen_single() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    cuda_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
        Some(16),
    );

    harness
        .dense_arena
        .get_record_seeker::<XorinVmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(all(feature = "cuda", feature = "rvr"))]
#[test]
fn test_xorin_preflight_replay_accepts_valid_transcript_and_rejects_unaligned_len() {
    let buffer_reg = 8usize;
    let input_reg = 16usize;
    let len_reg = 24usize;
    let buffer_ptr = 0x100u32;
    let input_ptr = 0x200u32;
    let len = KECCAK_RATE_BYTES;
    let xorin = Instruction::<F>::from_usize(
        XorinOpcode::XORIN.global_opcode(),
        [
            buffer_reg,
            input_reg,
            len_reg,
            RV64_REGISTER_AS as usize,
            RV64_MEMORY_AS as usize,
        ],
    );
    let instructions = [
        xorin,
        Instruction::from_usize(SystemOpcode::TERMINATE.global_opcode(), [0, 0, 0, 0, 0]),
    ];
    let program = Program::from_instructions(&instructions);
    let memory_config = openvm_circuit::arch::MemoryConfig::default();

    let block = |bytes: &[u8]| {
        std::array::from_fn(|i| u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
    };
    let register_block = |value: u64| block(&value.to_le_bytes());
    let mut memory_log = vec![
        PreflightMemoryEvent {
            timestamp: 1,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: buffer_reg as u32 / 2,
            value: register_block(buffer_ptr as u64),
        },
        PreflightMemoryEvent {
            timestamp: 2,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: input_reg as u32 / 2,
            value: register_block(input_ptr as u64),
        },
        PreflightMemoryEvent {
            timestamp: 3,
            address_space_and_kind: RV64_REGISTER_AS,
            pointer: len_reg as u32 / 2,
            value: register_block(len as u64),
        },
    ];
    let buffer_bytes = (0..len).map(|offset| offset as u8).collect::<Vec<_>>();
    let input_bytes = (0..len)
        .map(|offset| (offset as u8).wrapping_mul(17))
        .collect::<Vec<_>>();
    let num_blocks = len / MEMORY_BLOCK_BYTES;
    for i in 0..num_blocks {
        memory_log.push(PreflightMemoryEvent {
            timestamp: 4 + i as u32,
            address_space_and_kind: RV64_MEMORY_AS,
            pointer: buffer_ptr / 2 + (i * 4) as u32,
            value: block(&buffer_bytes[i * 8..][..8]),
        });
    }
    for i in 0..num_blocks {
        memory_log.push(PreflightMemoryEvent {
            timestamp: 4 + num_blocks as u32 + i as u32,
            address_space_and_kind: RV64_MEMORY_AS,
            pointer: input_ptr / 2 + (i * 4) as u32,
            value: block(&input_bytes[i * 8..][..8]),
        });
    }
    for i in 0..num_blocks {
        let output =
            std::array::from_fn::<_, 8, _>(|j| buffer_bytes[i * 8 + j] ^ input_bytes[i * 8 + j]);
        memory_log.push(PreflightMemoryEvent {
            timestamp: 4 + 2 * num_blocks as u32 + i as u32,
            address_space_and_kind: RV64_MEMORY_AS | PREFLIGHT_WRITE_BIT,
            pointer: buffer_ptr / 2 + (i * 4) as u32,
            value: block(&output),
        });
    }
    let final_timestamp = 4 + 3 * num_blocks as u32;
    let transcript = PreflightEventLog {
        program_log: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: final_timestamp,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: final_timestamp,
            },
        ],
        memory_log,
        initial_write_log: Vec::new(),
    };

    let tester = GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
    let chip = XorinVmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
    );

    let device_ctx = &tester.range_checker().device_ctx;
    let gpu_program = GpuPostflightProgram::upload(&program, &memory_config, device_ctx).unwrap();
    let (gpu_transcript, replay_plan) = gpu_program
        .upload_transcript(&transcript, PreflightEndpoint::Terminated)
        .unwrap();
    let _replay_ctx = chip
        .generate_proving_ctx_from_postflight(&gpu_program, &gpu_transcript, &replay_plan)
        .unwrap();
    assert_eq!(gpu_transcript.error_code().unwrap(), 0);

    let mut corrupt = transcript;
    corrupt.memory_log[2].value[0] = 7;
    let (gpu_corrupt, corrupt_plan) = gpu_program
        .upload_transcript(&corrupt, PreflightEndpoint::Terminated)
        .unwrap();
    let corrupt_chip = XorinVmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits() as u32,
    );
    corrupt_chip
        .generate_proving_ctx_from_postflight(&gpu_program, &gpu_corrupt, &corrupt_plan)
        .unwrap();
    assert_eq!(gpu_corrupt.error_code().unwrap(), 801);
}
