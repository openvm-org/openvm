use std::sync::Arc;

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
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use tiny_keccak::keccakf;

use super::KeccakfVmFiller;
use crate::keccakf::{air::KeccakfVmAir, KeccakfVmChip, KeccakfVmExecutor};

type F = BabyBear;
type Harness = TestChipHarness<F, KeccakfVmExecutor, KeccakfVmAir, KeccakfVmChip<F>>;

const KECCAK_STATE_BYTES: usize = 200;

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (KeccakfVmAir, KeccakfVmExecutor, KeccakfVmChip<F>) {
    let air = KeccakfVmAir::new(
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

    const MAX_TRACE_ROWS: usize = 4096;

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
    let rand_buffer = get_random_message(rng, KECCAK_STATE_BYTES);
    let mut rand_buffer_arr = [0u8; KECCAK_STATE_BYTES];
    rand_buffer_arr.copy_from_slice(&rand_buffer);

    let rd = gen_pointer(rng, 4);
    let buffer_ptr = gen_pointer(rng, KECCAK_STATE_BYTES);
    tester.write(1, rd, buffer_ptr.to_le_bytes().map(F::from_canonical_u8));
    let rand_buffer_arr_f = rand_buffer_arr.map(F::from_canonical_u8);

    for i in 0..(KECCAK_STATE_BYTES / 4) {
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

    let mut output_buffer = [F::from_canonical_u8(0); KECCAK_STATE_BYTES];

    for i in 0..(KECCAK_STATE_BYTES / 4) {
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

/// Test that verifies keccakf output against the tiny_keccak reference implementation.
/// This complements keccakf_chip_positive_tests by verifying execution correctness,
/// not just that traces pass AIR constraints.
#[test]
fn keccakf_output_verification_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // Test with multiple random inputs
    for _ in 0..5 {
        // Generate random 200-byte state
        let input_state: [u8; KECCAK_STATE_BYTES] = std::array::from_fn(|_| rng.gen::<u8>());

        // Compute expected output using tiny_keccak
        let mut expected_u64 = [0u64; 25];
        for (i, chunk) in input_state.chunks_exact(8).enumerate() {
            expected_u64[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        keccakf(&mut expected_u64);
        let mut expected_output = [0u8; KECCAK_STATE_BYTES];
        for (i, &val) in expected_u64.iter().enumerate() {
            expected_output[8 * i..8 * i + 8].copy_from_slice(&val.to_le_bytes());
        }

        // Setup memory pointers
        let rd = gen_pointer(&mut rng, 4);
        let buffer_ptr = gen_pointer(&mut rng, KECCAK_STATE_BYTES);
        tester.write(1, rd, buffer_ptr.to_le_bytes().map(F::from_canonical_u8));

        // Write input state to buffer
        for (i, chunk) in input_state.chunks(4).enumerate() {
            let chunk_f: [F; 4] = std::array::from_fn(|j| F::from_canonical_u8(chunk[j]));
            tester.write(2, buffer_ptr + i * 4, chunk_f);
        }

        // Execute keccakf
        tester.execute(
            &mut harness.executor,
            &mut harness.arena,
            &Instruction::from_usize(KeccakfOpcode::KECCAKF.global_opcode(), [rd, 0, 0, 1, 2]),
        );

        // Read output and verify
        let mut actual_output = [0u8; KECCAK_STATE_BYTES];
        for i in 0..(KECCAK_STATE_BYTES / 4) {
            let output_chunk: [F; 4] = tester.read(2, buffer_ptr + i * 4);
            for (j, &f) in output_chunk.iter().enumerate() {
                actual_output[4 * i + j] = f.as_canonical_u32() as u8;
            }
        }

        assert_eq!(
            actual_output, expected_output,
            "keccakf output mismatch for input state"
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
