use std::borrow::BorrowMut;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, MatrixRecordArena, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{pack_u8_block_value, MemoryBridge},
        SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    Rv64HintStoreOpcode::{self, *},
    MAX_HINT_BUFFER_DWORDS, MAX_HINT_BUFFER_DWORDS_BITS,
};
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng, RngCore};
#[cfg(feature = "cuda")]
use {
    crate::{Rv64HintStoreChipGpu, Rv64HintStoreLayout},
    openvm_circuit::arch::testing::{GpuChipTestBuilder, GpuTestChipHarness},
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
};

use super::{Rv64HintStoreAir, Rv64HintStoreChip, Rv64HintStoreCols, Rv64HintStoreExecutor};
use crate::{
    adapters::{u64_to_rv64_limbs, RV64_PTR_U16_LIMBS, U16_BITS},
    Rv64HintStoreFiller,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> =
    TestChipHarness<F, Rv64HintStoreExecutor, Rv64HintStoreAir, Rv64HintStoreChip<F>, RA>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv64HintStoreAir,
    Rv64HintStoreExecutor,
    Rv64HintStoreChip<F>,
) {
    let air = Rv64HintStoreAir::new(
        execution_bridge,
        memory_bridge,
        range_checker_chip.bus(),
        Rv64HintStoreOpcode::CLASS_OFFSET,
        address_bits,
    );
    let executor = Rv64HintStoreExecutor::new(address_bits, Rv64HintStoreOpcode::CLASS_OFFSET);
    let chip = Rv64HintStoreChip::<F>::new(
        Rv64HintStoreFiller::new(address_bits, range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness<RA: Arena>(tester: &mut VmChipTestBuilder<F>) -> Harness<RA> {
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        tester.range_checker().clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    Harness::<RA>::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64HintStoreOpcode,
) {
    let num_words = match opcode {
        HINT_STORED => 1,
        HINT_BUFFER => rng.random_range(1..28),
    } as u32;

    let a = if opcode == HINT_BUFFER {
        let a = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
        tester.write_bytes(
            RV64_REGISTER_AS as usize,
            a,
            u64_to_rv64_limbs(num_words.into()),
        );
        a
    } else {
        0
    };

    let mem_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        b,
        u64_to_rv64_limbs(mem_ptr.into()),
    );

    let num_bytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    let mut input = Vec::with_capacity(num_bytes);
    let mut hint_bytes = Vec::with_capacity(num_bytes);
    for _ in 0..num_words {
        let bytes = rng.next_u64().to_le_bytes();
        let data = bytes.map(F::from_u8);
        input.extend(data);
        hint_bytes.extend(bytes);
    }
    let streams = tester.streams_mut();
    streams.hint_stream.set_hint(hint_bytes);

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    for idx in 0..num_words as usize {
        let data = tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(
            RV64_MEMORY_AS as usize,
            mem_ptr as usize + idx * RV64_REGISTER_NUM_LIMBS,
        );

        let expected: [F; RV64_REGISTER_NUM_LIMBS] = input
            [idx * RV64_REGISTER_NUM_LIMBS..(idx + 1) * RV64_REGISTER_NUM_LIMBS]
            .try_into()
            .unwrap();
        assert_eq!(data, expected);
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_hintstore_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness(&mut tester);
    let num_ops: usize = 100;
    for _ in 0..num_ops {
        let opcode = if rng.random_bool(0.5) {
            HINT_STORED
        } else {
            HINT_BUFFER
        };
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
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

#[test]
#[should_panic(expected = "HintBufferTooLarge")]
fn test_hint_buffer_exceeds_max_words() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness::<MatrixRecordArena<F>>(&mut tester);

    let num_words = (MAX_HINT_BUFFER_DWORDS + 1) as u32;

    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        a,
        u64_to_rv64_limbs(num_words.into()),
    );

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        b,
        u64_to_rv64_limbs(mem_ptr.into()),
    );

    let hint_bytes = (0..num_words)
        .flat_map(|_| rng.next_u64().to_le_bytes())
        .collect();
    let streams = tester.streams_mut();
    streams.hint_stream.set_hint(hint_bytes);

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );
}

#[test]
#[should_panic(expected = "HintBufferZeroWords")]
fn test_hint_buffer_rejects_zero_words() {
    execute_invalid_hint_buffer(0, false);
}

#[test]
#[should_panic(expected = "num_words: 4294967297")]
fn test_hint_buffer_checks_full_rv64_count() {
    // The low 32 bits encode one word, so truncating before validation would accept this.
    execute_invalid_hint_buffer((1u64 << 32) | 1, true);
}

fn execute_invalid_hint_buffer(num_words: u64, provide_one_word: bool) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness::<MatrixRecordArena<F>>(&mut tester);

    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(RV64_REGISTER_AS as usize, a, u64_to_rv64_limbs(num_words));

    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        b,
        u64_to_rv64_limbs(gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u64),
    );
    if provide_one_word {
        tester
            .streams_mut()
            .hint_stream
            .extend([0; RV64_REGISTER_NUM_LIMBS]);
    }

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );
}

#[test]
fn test_hint_buffer_rem_words_range_check() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness(&mut tester);

    let num_words: u32 = 1;
    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        a,
        u64_to_rv64_limbs(num_words.into()),
    );

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        b,
        u64_to_rv64_limbs(mem_ptr.into()),
    );

    let hint_bytes = (0..num_words)
        .flat_map(|_| rng.next_u64().to_le_bytes())
        .collect();
    let streams = tester.streams_mut();
    streams.hint_stream.set_hint(hint_bytes);

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    let invalid_rem_words = 1 << MAX_HINT_BUFFER_DWORDS_BITS;
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut Rv64HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        // Set `rem_words` to the first value outside the allowed range.
        cols.rem_words = F::from_u32(invalid_rem_words);
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
fn test_hint_buffer_mem_ptr_range_check() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness(&mut tester);

    let num_words: u32 = 1;
    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        a,
        u64_to_rv64_limbs(num_words.into()),
    );

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(
        RV64_REGISTER_AS as usize,
        b,
        u64_to_rv64_limbs(mem_ptr.into()),
    );

    let hint_bytes = (0..num_words)
        .flat_map(|_| rng.next_u64().to_le_bytes())
        .collect();
    let streams = tester.streams_mut();
    streams.hint_stream.set_hint(hint_bytes);

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    let invalid_high_u16 = 1 << (tester.address_bits() - U16_BITS);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut Rv64HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        // Set the high u16 pointer cell to the first value outside the configured bound.
        cols.mem_ptr_limbs[RV64_PTR_U16_LIMBS - 1] = F::from_u32(invalid_high_u16);
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
#[should_panic(expected = "upper 4 bytes must be zero")]
fn test_hintstore_rs1_upper_bytes_non_zero() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness::<MatrixRecordArena<F>>(&mut tester);

    // Write b with a non-zero byte in the upper half; `mem_ptr_u64 >> 32` is then non-zero,
    // so the preflight executor must panic before it reaches the data write.
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    let mut mem_ptr_limbs = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
    mem_ptr_limbs[4] = F::from_u8(1);
    tester.write_bytes(RV64_REGISTER_AS as usize, b, mem_ptr_limbs);

    let data = rng.next_u64().to_le_bytes();
    let streams = tester.streams_mut();
    streams.hint_stream.set_hint(data.to_vec());

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_STORED.global_opcode(),
            [0, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );
}

#[allow(clippy::too_many_arguments)]
fn run_negative_hintstore_test(
    opcode: Rv64HintStoreOpcode,
    prank_data: Option<[F; BLOCK_FE_WIDTH]>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
    );

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut Rv64HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        if let Some(data) = prank_data {
            cols.data = data;
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
fn negative_hintstore_tests() {
    run_negative_hintstore_test(
        HINT_STORED,
        Some(pack_u8_block_value(
            &[92, 187, 45, 280, 17, 211, 64, 5].map(F::from_u32),
        )),
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let mut harness = create_harness::<MatrixRecordArena<F>>(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            HINT_STORED,
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
    Rv64HintStoreExecutor,
    Rv64HintStoreAir,
    Rv64HintStoreChipGpu,
    Rv64HintStoreChip<F>,
>;

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
        tester.address_bits(),
    );
    let gpu_chip = Rv64HintStoreChipGpu::new(
        tester.range_checker(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_hintstore_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

    let mut harness = create_cuda_harness(&tester);
    let num_ops = 50;
    for _ in 0..num_ops {
        let opcode = if rng.random_bool(0.5) {
            HINT_STORED
        } else {
            HINT_BUFFER
        };
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<_, Rv64HintStoreLayout>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
