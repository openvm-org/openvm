use std::{
    array::from_fn,
    sync::{Arc, Mutex},
};

use itertools::Itertools;
use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
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
use openvm_stark_backend::{
    interaction::{BusIndex, PermutationCheckBus},
    p3_field::{FieldAlgebra, PrimeField32},
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(feature = "cuda")]
use rand::Rng;
use tiny_keccak::keccakf;

use crate::{
    keccakf_op::{KeccakfExecutor, KeccakfOpAir, KeccakfOpChip},
    keccakf_perm::{KeccakfPermAir, KeccakfPermChip},
    KECCAK_WIDTH_BYTES,
};

type F = BabyBear;
/// Harness without KeccakfPeriphery*
type Harness<RA> = TestChipHarness<F, KeccakfExecutor, KeccakfOpAir, KeccakfOpChip<F>, RA>;
const MAX_TRACE_ROWS: usize = 4096;
const KECCAKF_STATE_BUS: BusIndex = 13;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (KeccakfOpAir, KeccakfExecutor, KeccakfOpChip<F>) {
    let executor = KeccakfExecutor::new(KeccakfOpcode::CLASS_OFFSET, address_bits);
    let empty_records = Arc::new(Mutex::new(Vec::new()));
    let op_air = KeccakfOpAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        PermutationCheckBus::new(KECCAKF_STATE_BUS),
        address_bits,
        KeccakfOpcode::CLASS_OFFSET,
    );
    let op_chip = KeccakfOpChip::new(bitwise_chip, address_bits, memory_helper, empty_records);
    (op_air, executor, op_chip)
}

struct TestHarness<RA> {
    harness: Harness<RA>,
    bitwise: (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
    perm: (KeccakfPermAir, KeccakfPermChip),
}

fn create_test_harness<RA: Arena>(tester: &mut VmChipTestBuilder<F>) -> TestHarness<RA> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (op_air, executor, op_chip) = create_harness_fields(
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );
    let shared_records = op_chip.shared_records.clone();

    let harness = Harness::with_capacity(executor, op_air, op_chip, MAX_TRACE_ROWS);

    let perm_air = KeccakfPermAir::new(op_air.keccakf_state_bus);
    let perm_chip = KeccakfPermChip::new(shared_records);

    TestHarness {
        harness,
        bitwise: (bitwise_chip.air, bitwise_chip),
        perm: (perm_air, perm_chip),
    }
}

fn set_and_execute_single_perm<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: KeccakfOpcode,
) {
    const MAX_LEN: usize = KECCAK_WIDTH_BYTES;
    let rand_buffer = get_random_message(rng, MAX_LEN);
    let mut rand_buffer_arr = [0u8; MAX_LEN];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

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

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, 0, 0, 1, 2]),
    );

    let mut output_buffer = [0u8; MAX_LEN];

    for i in 0..(MAX_LEN / 4) {
        let output_chunk: [F; 4] = tester.read(2, buffer_ptr + 4 * i);
        let output_chunk = output_chunk.map(|x| x.as_canonical_u32() as u8);
        output_buffer[4 * i..4 * i + 4].copy_from_slice(&output_chunk);
    }
    let mut state: [u64; 25] =
        from_fn(|i| u64::from_le_bytes(rand_buffer[8 * i..8 * i + 8].try_into().unwrap()));
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
    let num_ops: usize = 100;

    for _ in 0..num_ops {
        let mut rng = create_seeded_rng();
        let mut tester = VmChipTestBuilder::default();
        let TestHarness {
            mut harness,
            bitwise,
            perm,
        } = create_test_harness(&mut tester);

        set_and_execute_single_perm(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            KeccakfOpcode::KECCAKF,
        );

        let tester = tester
            .build()
            .load(harness)
            .load_periphery(perm)
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
use crate::{cuda::KeccakfVmChipGpu, keccakf_op::trace::KeccakfVmRecordMut};

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, KeccakfExecutor, KeccakfOpAir, KeccakfVmChipGpu, KeccakfVmChip<F>>;

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
    executor: &mut KeccakfExecutor,
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
