use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, ExecutionBridge, MatrixRecordArena, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_transpiler::{
    Rv64HintStoreOpcode::{self, *},
    MAX_HINT_BUFFER_DWORDS,
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
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use super::{Rv64HintStoreAir, Rv64HintStoreChip, Rv64HintStoreCols, Rv64HintStoreExecutor};
use crate::Rv64HintStoreFiller;

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> =
    TestChipHarness<F, Rv64HintStoreExecutor, Rv64HintStoreAir, Rv64HintStoreChip<F>, RA>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
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
        bitwise_chip.bus(),
        Rv64HintStoreOpcode::CLASS_OFFSET,
        address_bits,
    );
    let executor = Rv64HintStoreExecutor::new(address_bits, Rv64HintStoreOpcode::CLASS_OFFSET);
    let chip = Rv64HintStoreChip::<F>::new(
        Rv64HintStoreFiller::new(address_bits, bitwise_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness<RA: Arena>(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness<RA>,
    (
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let harness = Harness::<RA>::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

/// Convert a `u32` value to the 8-limb register representation (upper 4 limbs zero).
fn u32_to_rv64_limbs(x: u32) -> [F; RV64_REGISTER_NUM_LIMBS] {
    (x as u64).to_le_bytes().map(F::from_u8)
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
        tester.write(RV64_REGISTER_AS as usize, a, u32_to_rv64_limbs(num_words));
        a
    } else {
        0
    };

    let mem_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, b, u32_to_rv64_limbs(mem_ptr));

    let mut input = Vec::with_capacity(num_words as usize * RV64_REGISTER_NUM_LIMBS);
    for _ in 0..num_words {
        let data = rng.next_u64().to_le_bytes().map(F::from_u8);
        input.extend(data);
        tester.streams_mut().hint_stream.extend(data);
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    for idx in 0..num_words as usize {
        let data = tester.read::<RV64_REGISTER_NUM_LIMBS>(
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

    let (mut harness, bitwise) = create_harness(&mut tester);
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

#[test]
#[should_panic(expected = "HintBufferTooLarge")]
fn test_hint_buffer_exceeds_max_words() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, _bitwise) = create_harness::<MatrixRecordArena<F>>(&mut tester);

    let num_words = (MAX_HINT_BUFFER_DWORDS + 1) as u32;

    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, a, u32_to_rv64_limbs(num_words));

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, b, u32_to_rv64_limbs(mem_ptr));

    for _ in 0..num_words {
        let data = rng.next_u64().to_le_bytes().map(F::from_u8);
        tester.streams_mut().hint_stream.extend(data);
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

    let (mut harness, bitwise) = create_harness(&mut tester);

    let num_words: u32 = 1;
    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, a, u32_to_rv64_limbs(num_words));

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, b, u32_to_rv64_limbs(mem_ptr));

    for _ in 0..num_words {
        let data = rng.next_u64().to_le_bytes().map(F::from_u8);
        tester.streams_mut().hint_stream.extend(data);
    }

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut Rv64HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        // The AIR scales `rem_words_limbs[1]` by `1 << 6`, requiring the result to fit in a
        // byte (so `limb[1] < 4` under `MAX_HINT_BUFFER_DWORDS_BITS = 10`). Setting limb 1 to
        // 4 sends 256 to the byte-range lookup, which has no matching row. We compensate
        // `limb[0]` with `1 − 1024` in F so the composed `rem_words` stays at 1; otherwise
        // the end-row `assert_one(rem_words)` constraint would fail first and shadow the
        // interaction error we want to observe.
        cols.rem_words_limbs[1] = F::from_u32(4);
        cols.rem_words_limbs[0] = F::ONE - F::from_u32(1024);
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
fn test_hint_buffer_mem_ptr_range_check() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, bitwise) = create_harness(&mut tester);

    let num_words: u32 = 1;
    let a = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, a, u32_to_rv64_limbs(num_words));

    let mem_ptr = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS) as u32;
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    tester.write(RV64_REGISTER_AS as usize, b, u32_to_rv64_limbs(mem_ptr));

    for _ in 0..num_words {
        let data = rng.next_u64().to_le_bytes().map(F::from_u8);
        tester.streams_mut().hint_stream.extend(data);
    }

    tester.execute(
        &mut harness.executor,
        &mut harness.arena,
        &Instruction::from_usize(
            HINT_BUFFER.global_opcode(),
            [a, b, 0, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        ),
    );

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut Rv64HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        // For the default `pointer_max_bits = 29`, the AIR scales `mem_ptr_limbs[3]` by
        // `1 << 3`, which forces `mem_ptr_limbs[3] < 32`. Setting the limb to 100 sends a
        // scaled value of 800 to the byte-range bitwise lookup, which has no matching row.
        cols.mem_ptr_limbs[3] = F::from_u32(100);
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
#[should_panic(expected = "mem_ptr upper 4 bytes must be zero for hintstore")]
fn test_hintstore_rs1_upper_bytes_non_zero() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, _bitwise) = create_harness::<MatrixRecordArena<F>>(&mut tester);

    // Write b with a non-zero byte in the upper half; `mem_ptr_u64 >> 32` is then non-zero,
    // so the preflight executor must panic before it reaches the data write.
    let b = gen_pointer(&mut rng, RV64_REGISTER_NUM_LIMBS);
    let mut mem_ptr_limbs = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
    mem_ptr_limbs[4] = F::from_u8(1);
    tester.write(RV64_REGISTER_AS as usize, b, mem_ptr_limbs);

    let data = rng.next_u64().to_le_bytes().map(F::from_u8);
    tester.streams_mut().hint_stream.extend(data);

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
    prank_data: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&mut tester);

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
            cols.data = data.map(F::from_u32);
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
fn negative_hintstore_tests() {
    run_negative_hintstore_test(HINT_STORED, Some([92, 187, 45, 280, 17, 211, 64, 5]), true);
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

    let (mut harness, _) = create_harness::<MatrixRecordArena<F>>(&mut tester);

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
    Rv32HintStoreExecutor,
    Rv32HintStoreAir,
    Rv32HintStoreChipGpu,
    Rv32HintStoreChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    // getting bus from tester since `gpu_chip` and `air` must use the same bus
    let bitwise_bus = default_bitwise_lookup_bus();
    // creating a dummy chip for Cpu so we only count `add_count`s from GPU
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip.clone(),
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Rv32HintStoreChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_hintstore_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);
    let num_ops = 50;
    for _ in 0..num_ops {
        let opcode = if rng.random_bool(0.5) {
            HINT_STOREW
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
        .get_record_seeker::<_, Rv32HintStoreLayout>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
