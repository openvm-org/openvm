//! Tests for the BLAKE3 circuit extension.

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
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS},
    LocalOpcode,
};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use openvm_blake3_transpiler::Rv32Blake3Opcode::{self, *};

use super::{Blake3VmAir, Blake3VmExecutor, Blake3VmFiller};
use crate::{utils::blake3_hash_p3_full_blocks, Blake3VmChip, BLAKE3_BLOCK_BYTES};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> = TestChipHarness<F, Blake3VmExecutor, Blake3VmAir, Blake3VmChip<F>, RA>;

// ============================================================================
// Test Harness Setup
// ============================================================================

fn create_harness_fields(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (Blake3VmAir, Blake3VmExecutor, Blake3VmChip<F>) {
    let air = Blake3VmAir::new(
        execution_bridge,
        memory_bridge,
        bitwise_chip.bus(),
        address_bits,
        Rv32Blake3Opcode::CLASS_OFFSET,
    );
    let executor = Blake3VmExecutor::new(Rv32Blake3Opcode::CLASS_OFFSET, address_bits);
    let chip = Blake3VmChip::new(
        Blake3VmFiller::new(bitwise_chip, address_bits),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_harness<RA: Arena>(
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
        tester.execution_bridge(),
        tester.memory_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
        tester.address_bits(),
    );

    let harness = Harness::<RA>::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

// ============================================================================
// Test Execution Helper
// ============================================================================

/// Execute a BLAKE3 instruction and verify the output hash.
///
/// Handles memory setup with proper zero-padding to match circuit expectations.
#[allow(clippy::too_many_arguments)]
fn execute_blake3<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    message: Option<&[u8]>,
    len: Option<usize>,
    expected_output: Option<[u8; 32]>,
) {
    // Generate random message if not provided
    let len = len.unwrap_or(rng.gen_range(0..4096));
    let data = (0..len).map(|_| rng.gen::<u8>()).collect::<Vec<_>>();
    let message: &[u8] = message.unwrap_or(&data);
    let len = message.len();

    // Generate register and memory pointers
    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    let dst_ptr = gen_pointer(rng, 4);
    let src_ptr = gen_pointer(rng, 4);

    // Write register values
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    // Write message to memory with zero-padding to full blocks.
    // The circuit reads full 64-byte blocks, so padding must be consistent.
    let num_blocks = num_blake3_blocks(len);
    let full_len = num_blocks * BLAKE3_BLOCK_BYTES;
    let mut padded_msg = vec![0u8; full_len];
    padded_msg[..message.len()].copy_from_slice(message);

    for (i, chunk) in padded_msg.chunks_exact(4).enumerate() {
        let chunk: [u8; 4] = chunk.try_into().unwrap();
        tester.write(
            RV32_MEMORY_AS as usize,
            src_ptr + i * 4,
            chunk.map(F::from_canonical_u8),
        );
    }

    // Execute instruction
    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(BLAKE3.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    // Compute expected hash over full blocks (with zero padding) to match circuit
    let expected = expected_output.unwrap_or_else(|| {
        let mut out = [0u8; 32];
        out.copy_from_slice(&blake3_hash_p3_full_blocks(&padded_msg));
        out
    });

    assert_eq!(
        expected.map(F::from_canonical_u8),
        tester.read(RV32_MEMORY_AS as usize, dst_ptr),
        "BLAKE3 hash mismatch for input of {} bytes",
        len
    );
}

/// Calculate number of 64-byte blocks needed for input length.
#[inline]
fn num_blake3_blocks(len: usize) -> usize {
    if len == 0 {
        1
    } else {
        (len + BLAKE3_BLOCK_BYTES - 1) / BLAKE3_BLOCK_BYTES
    }
}

// ============================================================================
// Single Block Tests
// ============================================================================

#[test]
fn test_blake3_single_block() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // 32-byte message fits in single 64-byte block
    let msg = [0x42u8; 32];
    execute_blake3(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(&msg),
        Some(32),
        None,
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

// ============================================================================
// Multi-Block Tests
// ============================================================================

#[test]
fn test_blake3_two_blocks() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // 65-byte message requires 2 blocks (64 + 1)
    let msg: Vec<u8> = (0..65).map(|i| i as u8).collect();
    execute_blake3(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(&msg),
        Some(65),
        None,
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_blake3_three_blocks() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // 129-byte message requires 3 blocks (64 + 64 + 1)
    let msg: Vec<u8> = (0..129u8).collect();
    execute_blake3(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(&msg),
        Some(129),
        None,
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

// ============================================================================
// Randomized Tests
// ============================================================================

#[test]
fn test_blake3_random_inputs() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // Test 10 random inputs of varying sizes
    for _ in 0..10 {
        execute_blake3(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            None,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_blake3_length_boundaries() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_harness(&mut tester);

    // Test lengths at block boundaries and edge cases
    let test_lengths = [
        0,    // Empty input (special case)
        1,    // Minimum non-empty
        63,   // Just under block size
        64,   // Exact block size
        65,   // Just over block size (2 blocks)
        127,  // Just under 2 blocks
        128,  // Exact 2 blocks
        129,  // Just over 2 blocks (3 blocks)
        1023, // Just under 16 blocks
        1024, // Exact 16 blocks
        1025, // Just over 16 blocks
        8192, // Large input (128 blocks)
    ];

    for len in test_lengths {
        let msg: Vec<u8> = (0..len).map(|_| rng.gen::<u8>()).collect();
        execute_blake3(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            Some(&msg),
            Some(len),
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
