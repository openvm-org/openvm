use std::sync::Arc;

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use rand::Rng;

use super::KeccakfVmFiller;
use crate::keccakf_op::{air::KeccakfOpAir, KeccakfVmChip, KeccakfVmExecutor};

type F = BabyBear;
type Harness = TestChipHarness<F, KeccakfVmExecutor, KeccakfOpAir, KeccakfVmChip<F>>;
const MAX_TRACE_ROWS: usize = 4096;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (KeccakfOpAir, KeccakfVmExecutor, KeccakfVmChip<F>) {
    let air = KeccakfOpAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        address_bits,
        KeccakfOpcode::CLASS_OFFSET,
    );

    let executor = KeccakfVmExecutor::new(KeccakfOpcode::CLASS_OFFSET, address_bits);
    let chip = KeccakfVmChip::new(
        KeccakfVmFiller::new(bitwise_chip, address_bits),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
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
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );

    let harness = Harness::with_capacity(executor, air, chip, MAX_TRACE_ROWS);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: KeccakfOpcode,
) {
    const MAX_LEN: usize = 200;
    let rand_buffer = get_random_message(rng, MAX_LEN);
    let mut rand_buffer_arr = [0u8; MAX_LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    use openvm_circuit::arch::testing::memory::gen_pointer;
    let rd = gen_pointer(rng, 4);
    let buffer_ptr = gen_pointer(rng, MAX_LEN);
    tester.write(1, rd, buffer_ptr.to_le_bytes().map(F::from_canonical_u8));
    let rand_buffer_arr_f = rand_buffer_arr.map(F::from_canonical_u8);

    for i in 0..(MAX_LEN / 4) {
        let buffer_chunk: [F; 4] = rand_buffer_arr_f[4 * i..4 * i + 4]
            .try_into()
            .expect("slice has length 4");
        tester.write(2, buffer_ptr + 4 * i, buffer_chunk);
    }

    // Safety note: If you would like to increase this further, make sure to count if it will
    // exceed the MAX_TRACE_ROWS of the test harness configs or not
    // currently it does not because 100 * 24 < 4096
    for _ in 0..100 {
        tester.execute(
            executor,
            arena,
            &Instruction::from_usize(opcode.global_opcode(), [rd, 0, 0, 1, 2]),
        );
    }

    let mut output_buffer = [F::from_canonical_u8(0); MAX_LEN];

    for i in 0..(MAX_LEN / 4) {
        let output_chunk: [F; 4] = tester.read(2, buffer_ptr + 4 * i);
        output_buffer[4 * i..4 * i + 4].copy_from_slice(&output_chunk);
    }
}

#[test]
fn keccakf_chip_positive_tests() {
    let num_ops: usize = 10;

    for _ in 0..num_ops {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let (mut harness, bitwise) = create_test_harness(&mut tester);

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            KeccakfOpcode::KECCAKF,
        );

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(bitwise)
            .finalize();
        tester.simple_test().expect("Verification failed");
    }
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
use crate::{cuda::KeccakfVmChipGpu, keccakf::trace::KeccakfVmRecordMut};

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, KeccakfVmExecutor, KeccakfOpAir, KeccakfVmChipGpu, KeccakfVmChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );

    let gpu_chip = KeccakfVmChipGpu::new(
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
    executor: &mut KeccakfVmExecutor,
    arena: &mut DenseRecordArena,
    rng: &mut StdRng,
) {
    use openvm_circuit::arch::testing::memory::gen_pointer;

    const KECCAK_STATE_BYTES: usize = 200;

    let buffer_reg = gen_pointer(rng, 4);
    let buffer_ptr = gen_pointer(rng, KECCAK_STATE_BYTES);

    tester.write(
        1,
        buffer_reg,
        buffer_ptr.to_le_bytes().map(F::from_canonical_u8),
    );

    let state_data: Vec<u8> = (0..KECCAK_STATE_BYTES).map(|_| rng.gen()).collect();
    for (i, chunk) in state_data.chunks(4).enumerate() {
        let mut word = [F::ZERO; 4];
        for (j, &byte) in chunk.iter().enumerate() {
            word[j] = F::from_canonical_u8(byte);
        }
        tester.write(2, buffer_ptr + i * 4, word);
    }

    let instruction = Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [buffer_reg, 0, 0, 1, 2],
    );

    tester.execute(executor, arena, &instruction);
}

#[cfg(feature = "cuda")]
#[test]
fn test_keccakf_cuda_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    let num_ops: usize = 3;
    for _ in 0..num_ops {
        cuda_set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<KeccakfVmRecordMut, _>()
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
fn test_keccakf_cuda_tracegen_single() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    cuda_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        &mut rng,
    );

    harness
        .dense_arena
        .get_record_seeker::<KeccakfVmRecordMut, _>()
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
fn test_keccakf_cuda_tracegen_zero_state() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    use openvm_circuit::arch::testing::memory::gen_pointer;

    const KECCAK_STATE_BYTES: usize = 200;

    let buffer_reg = gen_pointer(&mut rng, 4);
    let buffer_ptr = gen_pointer(&mut rng, KECCAK_STATE_BYTES);

    tester.write(
        1,
        buffer_reg,
        buffer_ptr.to_le_bytes().map(F::from_canonical_u8),
    );

    for i in 0..(KECCAK_STATE_BYTES / 4) {
        tester.write(2, buffer_ptr + i * 4, [F::ZERO; 4]);
    }

    let instruction = Instruction::from_usize(
        KeccakfOpcode::KECCAKF.global_opcode(),
        [buffer_reg, 0, 0, 1, 2],
    );

    tester.execute(
        &mut harness.executor,
        &mut harness.dense_arena,
        &instruction,
    );

    harness
        .dense_arena
        .get_record_seeker::<KeccakfVmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
