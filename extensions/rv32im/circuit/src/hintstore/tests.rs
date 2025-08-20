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
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::Rv32HintStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_field::FieldAlgebra,
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
    crate::{Rv32HintStoreChipGpu, Rv32HintStoreLayout},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use super::{Rv32HintStoreAir, Rv32HintStoreChip, Rv32HintStoreCols, Rv32HintStoreExecutor};
use crate::{test_utils::get_verification_error, Rv32HintStoreFiller};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> =
    TestChipHarness<F, Rv32HintStoreExecutor, Rv32HintStoreAir, Rv32HintStoreChip<F>, RA>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Rv32HintStoreAir,
    Rv32HintStoreExecutor,
    Rv32HintStoreChip<F>,
) {
    let air = Rv32HintStoreAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        Rv32HintStoreOpcode::CLASS_OFFSET,
        address_bits,
    );
    let executor = Rv32HintStoreExecutor::new(address_bits, Rv32HintStoreOpcode::CLASS_OFFSET);
    let chip = Rv32HintStoreChip::<F>::new(
        Rv32HintStoreFiller::new(address_bits, bitwise_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness<RA: Arena>(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness<RA>,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
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

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv32HintStoreOpcode,
) {
    let num_words = match opcode {
        HINT_STOREW => 1,
        HINT_BUFFER => rng.gen_range(1..28),
    } as u32;

    let a = if opcode == HINT_BUFFER {
        let a = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
        tester.write(
            RV32_REGISTER_AS as usize,
            a,
            num_words.to_le_bytes().map(F::from_canonical_u8),
        );
        a
    } else {
        0
    };

    let mem_ptr = gen_pointer(rng, 4) as u32;
    let b = gen_pointer(rng, RV32_REGISTER_NUM_LIMBS);
    tester.write(1, b, mem_ptr.to_le_bytes().map(F::from_canonical_u8));

    let mut input = Vec::with_capacity(num_words as usize * 4);
    for _ in 0..num_words {
        let data = rng.next_u32().to_le_bytes().map(F::from_canonical_u8);
        input.extend(data);
        tester.streams_mut().hint_stream.extend(data);
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [a, b, 0, RV32_REGISTER_AS as usize, RV32_MEMORY_AS as usize],
        ),
    );

    for idx in 0..num_words as usize {
        let data = tester.read::<4>(RV32_MEMORY_AS as usize, mem_ptr as usize + idx * 4);

        let expected: [F; 4] = input[idx * 4..(idx + 1) * 4].try_into().unwrap();
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
        let opcode = if rng.gen_bool(0.5) {
            HINT_STOREW
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

#[allow(clippy::too_many_arguments)]
fn run_negative_hintstore_test(
    opcode: Rv32HintStoreOpcode,
    prank_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    interaction_error: bool,
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
        let mut trace_row = trace.row_slice(0).to_vec();
        let cols: &mut Rv32HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        if let Some(data) = prank_data {
            cols.data = data.map(F::from_canonical_u32);
        }
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn negative_hintstore_tests() {
    run_negative_hintstore_test(HINT_STOREW, Some([92, 187, 45, 280]), true);
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
            HINT_STOREW,
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
        let opcode = if rng.gen_bool(0.5) {
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
