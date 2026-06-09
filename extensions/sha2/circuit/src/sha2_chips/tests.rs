use std::sync::{Arc, Mutex};

use hex::FromHex;
use itertools::Itertools;
use openvm_circuit::{
    arch::{
        testing::{
            memory::{gen_pointer, gen_register_pointer},
            TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
        },
        Arena, MatrixRecordArena, PreflightExecutor,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
    utils::get_random_message,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerChip},
    ColumnsAir, SubAir,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_sha2_air::{
    get_flag_pt_array, set_arrayview_from_u32_slice, small_sig0, small_sig1, word_into_bits,
    word_into_u16_limbs, word_into_u8_limbs, Sha256Config, Sha2BlockHasherSubAir,
    Sha2BlockHasherSubairConfig, Sha2RoundColsRefMut, Sha384Config, Sha512Config, SHA256_K,
};
use openvm_sha2_transpiler::Rv64Sha2Opcode;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    utils::disable_debug_builder,
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{Sha2BlockHasherChipGpu, Sha2MainChipGpu, Sha2RecordMut},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
    openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupChipGPU,
};

use crate::{
    add_padding_to_message, read_slice_from_memory, write_slice_to_memory, Sha2BlockHasherChip,
    Sha2BlockHasherDigestColsRefMut, Sha2BlockHasherVmAir, Sha2Config, Sha2MainAir, Sha2MainChip,
    Sha2VmExecutor,
};

const SHA2_BUS_IDX: BusIndex = 28;
const SUBAIR_BUS_IDX: BusIndex = 29;
type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA, C> = TestChipHarness<F, Sha2VmExecutor<C>, Sha2MainAir<C>, Sha2MainChip<F, C>, RA>;

fn create_harness_fields<C: Sha2Config>(
    system_port: SystemPort,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
    pointer_max_bits: usize,
) -> (Sha2MainAir<C>, Sha2VmExecutor<C>, Sha2MainChip<F, C>) {
    let executor = Sha2VmExecutor::<C>::new(Rv64Sha2Opcode::CLASS_OFFSET, pointer_max_bits);
    let empty_records = Arc::new(Mutex::new(None));
    let main_chip = Sha2MainChip::new(
        empty_records.clone(),
        range_checker_chip.clone(),
        pointer_max_bits,
        memory_helper,
    );
    let main_air = Sha2MainAir::new(
        system_port,
        range_checker_chip.bus(),
        pointer_max_bits,
        SHA2_BUS_IDX,
        Rv64Sha2Opcode::CLASS_OFFSET,
    );
    (main_air, executor, main_chip)
}

struct TestHarness<RA, C: Sha2Config> {
    harness: Harness<RA, C>,
    bitwise: (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
    block_hasher: (Sha2BlockHasherVmAir<C>, Sha2BlockHasherChip<F, C>),
}

fn create_test_harness<RA: Arena, C: Sha2Config>(
    tester: &mut VmChipTestBuilder<F>,
) -> TestHarness<RA, C> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, main_chip) = create_harness_fields(
        tester.system_port(),
        tester.range_checker(),
        tester.memory_helper(),
        tester.address_bits(),
    );

    let shared_records = main_chip.records.clone();

    let harness = Harness::<RA, C>::with_capacity(executor, air, main_chip, MAX_INS_CAPACITY);

    let range_checker = tester.range_checker();
    let block_hasher_air = Sha2BlockHasherVmAir::new(
        bitwise_chip.bus(),
        range_checker.bus(),
        SUBAIR_BUS_IDX,
        SHA2_BUS_IDX,
    );
    let block_hasher_chip = Sha2BlockHasherChip::new(
        bitwise_chip.clone(),
        range_checker,
        tester.address_bits(),
        tester.memory_helper(),
        shared_records,
    );

    TestHarness {
        harness,
        bitwise: (bitwise_chip.air, bitwise_chip),
        block_hasher: (block_hasher_air, block_hasher_chip),
    }
}

// execute one SHA2_UPDATE instruction
#[allow(clippy::too_many_arguments)]
fn set_and_execute_single_block<RA: Arena, C: Sha2Config, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64Sha2Opcode,
    message: Option<&[u8]>,
    prev_state: Option<&[u8]>,
) {
    let rd = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs1 = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2 = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    let dst_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let state_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let input_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(1, rd, (dst_ptr as u64).to_le_bytes().map(F::from_u8));
    tester.write_bytes(1, rs1, (state_ptr as u64).to_le_bytes().map(F::from_u8));
    tester.write_bytes(1, rs2, (input_ptr as u64).to_le_bytes().map(F::from_u8));

    let default_message = get_random_message(rng, C::BLOCK_U8S);
    let message = message.unwrap_or(&default_message);
    assert!(message.len() == C::BLOCK_U8S);
    write_slice_to_memory(tester, message, input_ptr);

    let default_prev_state = get_random_message(rng, C::STATE_BYTES);
    let prev_state = prev_state.unwrap_or(&default_prev_state);
    assert!(prev_state.len() == C::STATE_BYTES);
    write_slice_to_memory(tester, prev_state, state_ptr);

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let mut state = prev_state.to_vec();
    C::compress(&mut state, message);
    let expected_output = state.iter().cloned().map(F::from_u8).collect_vec();

    assert_eq!(
        expected_output,
        read_slice_from_memory(tester, dst_ptr, C::STATE_BYTES)
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS - Single Block Hash
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
// Test a single block hash
fn rand_sha2_single_block_test<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        block_hasher,
    } = create_test_harness::<MatrixRecordArena<F>, C>(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute_single_block::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(block_hasher)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_single_block_test() {
    rand_sha2_single_block_test::<Sha256Config>();
}

#[test]
fn rand_sha512_single_block_test() {
    rand_sha2_single_block_test::<Sha512Config>();
}

#[test]
fn rand_sha384_single_block_test() {
    rand_sha2_single_block_test::<Sha384Config>();
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS - Multi Block Hash
///
/// Execute multiple SHA2_UPDATE instructions to hash an entire message
///////////////////////////////////////////////////////////////////////////////////////
#[allow(clippy::too_many_arguments)]
fn set_and_execute_full_message<RA: Arena, C: Sha2Config + 'static, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv64Sha2Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let rd = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs1 = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2 = gen_register_pointer(rng, RV64_REGISTER_NUM_LIMBS);

    let state_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let dst_ptr = state_ptr;
    let input_ptr = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    tester.write_bytes(1, rd, (dst_ptr as u64).to_le_bytes().map(F::from_u8));
    tester.write_bytes(1, rs1, (state_ptr as u64).to_le_bytes().map(F::from_u8));
    tester.write_bytes(1, rs2, (input_ptr as u64).to_le_bytes().map(F::from_u8));

    // initial state as little-endian words
    let initial_state: Vec<u8> = C::get_h()
        .iter()
        .cloned()
        .flat_map(|x| word_into_u8_limbs::<C>(x).into_iter())
        .map(|x| x.try_into().unwrap())
        .collect_vec();

    assert!(initial_state.len() == C::STATE_BYTES);
    write_slice_to_memory(tester, &initial_state, state_ptr);

    let len = len.unwrap_or(rng.random_range(1..3000));
    let default_message = get_random_message(rng, len);
    let message = message.map(|x| x.to_vec()).unwrap_or(default_message);

    // C::hash() returns big-endian words.
    // We want little-endian words so we can compare to our final state (which is in little-endian
    // words)
    let expected_output = C::hash(&message)
        .chunks_exact(C::WORD_U8S)
        .flat_map(|word| word.iter().rev().copied())
        .collect_vec();

    let padded_message = add_padding_to_message::<C>(message);

    // run SHA2_UPDATE as many times as needed to hash the entire message
    padded_message
        .chunks_exact(C::BLOCK_BYTES)
        .for_each(|block| {
            write_slice_to_memory(tester, block, input_ptr);

            tester.execute(
                executor,
                arena,
                &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
            );
        });

    let output = read_slice_from_memory(tester, dst_ptr, C::DIGEST_BYTES)
        .into_iter()
        .map(|x| x.as_canonical_u32() as u8)
        .collect_vec();

    assert_eq!(expected_output, output);
}

// Test a single block hash
fn rand_sha2_multi_block_test<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        block_hasher,
    } = create_test_harness::<_, C>(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(block_hasher)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_multi_block_test() {
    rand_sha2_multi_block_test::<Sha256Config>();
}

#[test]
fn rand_sha512_multi_block_test() {
    rand_sha2_multi_block_test::<Sha512Config>();
}

// Note that this test is distinct from rand_sha512_multi_block_test() because this one uses the
// initial hash state for SHA384 instead of SHA512.
#[test]
fn rand_sha384_multi_block_test() {
    rand_sha2_multi_block_test::<Sha384Config>();
}

///////////////////////////////////////////////////////////////////////////////////////
/// EDGE TESTS - Edge Case Input Lengths
///
/// Test the hash function with various input lengths.
///////////////////////////////////////////////////////////////////////////////////////
fn sha2_edge_test_lengths<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        block_hasher,
    } = create_test_harness::<_, C>(&mut tester);

    // inputs of various number of blocks
    const TEST_VECTORS: [&str; 4] = [
        "",
        "98c1c0bdb7d5fea9a88859f06c6c439f",
        "5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c",
        "9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e",
    ];

    for input in TEST_VECTORS.iter() {
        let input = Vec::from_hex(input).unwrap();

        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            Some(&input),
            None,
        );
    }

    // check every possible input length modulo block size
    for i in (C::BLOCK_BYTES + 1)..=(2 * C::BLOCK_BYTES) {
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            None,
            Some(i),
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(block_hasher)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn sha256_edge_test_lengths() {
    sha2_edge_test_lengths::<Sha256Config>();
}

#[test]
fn sha512_edge_test_lengths() {
    sha2_edge_test_lengths::<Sha512Config>();
}

#[test]
fn sha384_edge_test_lengths() {
    sha2_edge_test_lengths::<Sha384Config>();
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
fn execute_roundtrip_sanity_test<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness { mut harness, .. } =
        create_test_harness::<MatrixRecordArena<F>, C>(&mut tester);

    // let num_tests: usize = 10;
    let num_tests: usize = 1;
    for _ in 0..num_tests {
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            None,
            // None,
            Some(0),
        );
    }
}

#[test]
fn execute_roundtrip_sanity_test_sha256() {
    execute_roundtrip_sanity_test::<Sha256Config>();
}

#[test]
fn execute_roundtrip_sanity_test_sha512() {
    execute_roundtrip_sanity_test::<Sha512Config>();
}
#[test]
fn execute_roundtrip_sanity_test_sha384() {
    execute_roundtrip_sanity_test::<Sha384Config>();
}

#[test]
fn sha256_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = Sha256Config::hash(input);
    let expected: [u8; 32] = [
        99, 196, 61, 185, 226, 212, 131, 80, 154, 248, 97, 108, 157, 55, 200, 226, 160, 73, 207,
        46, 245, 169, 94, 255, 42, 136, 193, 15, 40, 133, 173, 22,
    ];
    assert_eq!(output, expected);
}

#[test]
fn sha512_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = Sha512Config::hash(input);
    let expected: [u8; 64] = [
        0, 8, 195, 142, 70, 71, 97, 208, 132, 132, 243, 53, 179, 186, 8, 162, 71, 75, 126, 21, 130,
        203, 245, 126, 207, 65, 119, 60, 64, 79, 200, 2, 194, 17, 189, 137, 164, 213, 107, 197,
        152, 11, 242, 165, 146, 80, 96, 105, 249, 27, 139, 14, 244, 21, 118, 31, 94, 87, 32, 145,
        149, 98, 235, 75,
    ];
    assert_eq!(output, expected);
}

#[test]
fn sha384_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = Sha384Config::hash(input);
    let expected: [u8; 48] = [
        134, 227, 167, 229, 35, 110, 115, 174, 10, 27, 197, 116, 56, 144, 150, 36, 152, 190, 212,
        120, 26, 243, 125, 4, 2, 60, 164, 195, 218, 219, 255, 143, 240, 75, 158, 126, 102, 105, 8,
        202, 142, 240, 230, 161, 162, 152, 111, 71,
    ];
    assert_eq!(output, expected);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// This tests a soundness bug that was found at one point in our implementation.
///////////////////////////////////////////////////////////////////////////////////////
fn negative_sha2_test_bad_final_hash<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let TestHarness {
        mut harness,
        bitwise,
        block_hasher,
    } = create_test_harness::<MatrixRecordArena<F>, C>(&mut tester);

    let num_ops: usize = 1;
    for _ in 0..num_ops {
        set_and_execute_single_block::<_, C, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            C::OPCODE,
            None,
            None,
        );
    }

    // Set the final_hash to all zeros
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        trace.row_chunks_exact_mut(1).for_each(|row| {
            let mut row_slice = row.row_slice(0).unwrap().to_vec();
            let mut cols = Sha2BlockHasherDigestColsRefMut::from::<C>(
                &mut row_slice[..C::BLOCK_HASHER_DIGEST_WIDTH],
            );
            if cols.inner.flags.is_digest_row.is_one() {
                for i in 0..C::HASH_WORDS {
                    for j in 0..C::WORD_U16S {
                        cols.inner.final_hash[[i, j]] = F::ZERO;
                    }
                }
                row.values.copy_from_slice(&row_slice);
            }
        });
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load(harness)
        .load_periphery_and_prank_trace(block_hasher, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn negative_sha256_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha256Config>();
}

#[test]
fn negative_sha512_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha512Config>();
}

#[test]
fn negative_sha384_test_bad_final_hash() {
    negative_sha2_test_bad_final_hash::<Sha384Config>();
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS - Missing Terminal Padding Row Enforcement
//
// Tests that the Sha2BlockHasherSubAir rejects traces where the last row is not a
// padding row. Without the `when_last_row().assert_one(is_padding_row)` constraint,
// a malicious prover could submit a truncated 2-row trace containing only round rows
// with all-zero working state and W_t = -K_t, bypassing all digest, finalization,
// and hash-chaining constraints.
//
// To verify that this test catches the vulnerability, temporarily remove the
// `when_last_row()` assertion in `Sha2BlockHasherSubAir::eval_transitions` and
// confirm the test fails (i.e., the prank trace is accepted).
//////////////////////////////////////////////////////////////////////////////////////

/// Standalone test wrapper that exposes `Sha2BlockHasherSubAir` as a top-level `Air`.
struct Sha2SubAirTestWrapper<C: Sha2BlockHasherSubairConfig> {
    sub_air: Sha2BlockHasherSubAir<C>,
}

impl<C: Sha2BlockHasherSubairConfig> ColumnsAir for Sha2SubAirTestWrapper<C> {}

impl<F: Field, C: Sha2BlockHasherSubairConfig> BaseAirWithPublicValues<F>
    for Sha2SubAirTestWrapper<C>
{
}
impl<F: Field, C: Sha2BlockHasherSubairConfig> PartitionedBaseAir<F> for Sha2SubAirTestWrapper<C> {}
impl<F: Field, C: Sha2BlockHasherSubairConfig> BaseAir<F> for Sha2SubAirTestWrapper<C> {
    fn width(&self) -> usize {
        C::SUBAIR_WIDTH
    }
}

impl<AB: InteractionBuilder, C: Sha2BlockHasherSubairConfig> Air<AB> for Sha2SubAirTestWrapper<C> {
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

/// Builds a malicious 2-row trace for `Sha2BlockHasherSubAir<Sha256Config>` where both
/// rows are round rows (row_idx 0 and 1) and no digest or padding row ever appears.
///
/// All working variables (a, e) are zero with W_t = -K_t mod 2^32, making compression
/// constraints trivially satisfied. Helper columns are set to satisfy the unconditional
/// message schedule constraints.
fn build_sha256_round_only_trace() -> RowMajorMatrix<F> {
    type C = Sha256Config;
    let width = C::SUBAIR_WIDTH;
    let round_w = C::SUBAIR_ROUND_WIDTH;
    let mut values = vec![F::ZERO; 2 * width];
    let encoder = Encoder::new(C::ROWS_PER_BLOCK + 1, 2, false);

    let neg_k: Vec<u32> = (0..8).map(|i| 0u32.wrapping_sub(SHA256_K[i])).collect();
    let limb_pair = |w: u32| -> [u32; 2] {
        let l = word_into_u16_limbs::<C>(w);
        [l[0], l[1]]
    };

    // Phase 1: flags, W bits, and work-variable carries. With (a, e) all zero and
    // W = -K mod 2^32, the work-var addition reduces to K + W = 0 per limb, which
    // (since every SHA-256 K has non-zero low 16 bits) gives carry_a = carry_e = [1, 1].
    for row in 0..2 {
        let start = row * width;
        let mut cols = Sha2RoundColsRefMut::<F>::from::<C>(&mut values[start..start + round_w]);
        *cols.flags.is_round_row = F::ONE;
        *cols.flags.is_first_4_rows = F::ONE;
        set_arrayview_from_u32_slice(&mut cols.flags.row_idx, get_flag_pt_array(&encoder, row));
        *cols.flags.global_block_idx = F::ONE;
        for i in 0..C::ROUNDS_PER_ROW {
            set_arrayview_from_u32_slice(
                &mut cols.message_schedule.w.row_mut(i),
                word_into_bits::<C>(neg_k[row * 4 + i]),
            );
            for j in 0..C::WORD_U16S {
                cols.work_vars.carry_a[[i, j]] = F::ONE;
                cols.work_vars.carry_e[[i, j]] = F::ONE;
            }
        }
    }

    // Phases 2-4: helper-column values. Each (local, next) pair generates the same
    // pattern, just with the rows swapped — one iteration covers the local->next
    // transition (row 0 -> row 1) and the other covers the wrap-around (row 1 -> row 0).
    for local in 0..2 {
        let next = (local + 1) % 2;
        // Concatenated W array seen by the local row: w[0..4] = local.w, w[4..8] = next.w.
        let w: Vec<u32> = (0..8).map(|k| neg_k[(local * 4 + k) % 8]).collect();

        // Constraints that bind the *next* row's helper columns:
        //   next.w_3[i]       = u16_limbs(w[i+1])              for i in 0..3
        //   next.intermed_4[i] = u16_limbs(w[i]) + u16_limbs(sig0(w[i+1]))
        {
            let start = next * width;
            let mut cols = Sha2RoundColsRefMut::<F>::from::<C>(&mut values[start..start + round_w]);
            for i in 0..C::ROUNDS_PER_ROW {
                if i < C::ROUNDS_PER_ROW - 1 {
                    set_arrayview_from_u32_slice(
                        &mut cols.schedule_helper.w_3.row_mut(i),
                        word_into_u16_limbs::<C>(w[i + 1]),
                    );
                }
                let w_limbs = limb_pair(w[i]);
                let sig_limbs = limb_pair(small_sig0::<C>(w[i + 1]));
                for j in 0..C::WORD_U16S {
                    cols.schedule_helper.intermed_4[[i, j]] =
                        F::from_u32(w_limbs[j] + sig_limbs[j]);
                }
            }
        }

        // Constraint that binds the *local* row's intermed_12 (carry_or_buffer left at 0):
        //   sig1(w[i+2]) + w_7 + intermed_12 == w[i+4]   (per u16 limb)
        // where w_7 = local.w_3[i] for i < 3, else u16_limbs(w[i-3]).
        // local.w_3[i] was constrained by the *other* iteration to equal u16_limbs(w[i+5])
        // here, so we can read w[i+5] directly instead of round-tripping through the trace.
        {
            let start = local * width;
            let mut cols = Sha2RoundColsRefMut::<F>::from::<C>(&mut values[start..start + round_w]);
            for i in 0..C::ROUNDS_PER_ROW {
                let w_t = limb_pair(w[i + 4]);
                let sig1 = limb_pair(small_sig1::<C>(w[i + 2]));
                let w_7 = if i < 3 {
                    limb_pair(w[i + 5])
                } else {
                    limb_pair(w[i - 3])
                };
                for j in 0..C::WORD_U16S {
                    cols.schedule_helper.intermed_12[[i, j]] =
                        F::from_u32(w_t[j]) - F::from_u32(sig1[j]) - F::from_u32(w_7[j]);
                }
            }
        }
    }

    RowMajorMatrix::new(values, width)
}

#[test]
fn negative_sha256_round_only_trace_rejected() {
    disable_debug_builder();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));
    let range_bus = openvm_circuit::arch::testing::default_var_range_checker_bus();
    let range_checker_chip = Arc::new(VariableRangeCheckerChip::new(range_bus));

    // 2 rows × 4 rounds × 2 u16-limbs = 16 range checks, all (1, 1)
    for _ in 0..16 {
        bitwise_chip.request_range(1, 1);
    }

    let trace = build_sha256_round_only_trace();
    let air = Sha2SubAirTestWrapper {
        sub_air: Sha2BlockHasherSubAir::<Sha256Config>::new(bitwise_bus, range_bus, SUBAIR_BUS_IDX),
    };
    let air_ctx = AirProvingContext::simple_no_pis(trace);

    let tester = VmChipTestBuilder::default()
        .build()
        .load_air_proving_ctx((Arc::new(air) as AirRef<_>, air_ctx))
        .load_periphery((bitwise_chip.air, bitwise_chip))
        .load_periphery((range_checker_chip.air, range_checker_chip))
        .finalize();

    tester
        .simple_test()
        .expect_err("Round-only trace should be rejected by when_last_row constraint");
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type Sha2GpuTestChip<C> = GpuTestChipHarness<
    F,
    Sha2VmExecutor<C>,
    Sha2MainAir<C>,
    Sha2MainChipGpu<C>,
    Sha2MainChip<F, C>,
>;

#[cfg(feature = "cuda")]
struct GpuHarness<C: Sha2Config> {
    pub main: Sha2GpuTestChip<C>,
    block_air: Sha2BlockHasherVmAir<C>,
    block_gpu: Sha2BlockHasherChipGpu<C>,
    block_cpu: Sha2BlockHasherChip<F, C>,
    bitwise_air: BitwiseOperationLookupAir<RV64_BYTE_BITS>,
    bitwise_gpu: Arc<BitwiseOperationLookupChipGPU<8>>,
}

#[cfg(feature = "cuda")]
fn create_cuda_harness<C: Sha2Config>(tester: &GpuChipTestBuilder) -> GpuHarness<C> {
    const GPU_MAX_INS_CAPACITY: usize = 8192;
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (main_air, main_executor, main_chip) = create_harness_fields(
        tester.system_port(),
        dummy_range_checker_chip.clone(),
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let block_hasher_air = Sha2BlockHasherVmAir::new(
        bitwise_bus,
        dummy_range_checker_chip.bus(),
        SUBAIR_BUS_IDX,
        SHA2_BUS_IDX,
    );
    let block_hasher_chip = Sha2BlockHasherChip::new(
        dummy_bitwise_chip.clone(),
        dummy_range_checker_chip,
        tester.address_bits(),
        tester.dummy_memory_helper(),
        main_chip.records.clone(),
    );

    let shared_records_gpu = Arc::new(Mutex::new(None));
    let main_gpu_chip = Sha2MainChipGpu::new(
        shared_records_gpu.clone(),
        tester.range_checker(),
        tester.address_bits() as u32,
        tester.timestamp_max_bits() as u32,
    );

    let block_gpu_chip = Sha2BlockHasherChipGpu::new(
        shared_records_gpu.clone(),
        tester.bitwise_op_lookup(),
        tester.range_checker(),
    );

    let bitwise_gpu = tester.bitwise_op_lookup();
    let bitwise_air = BitwiseOperationLookupAir::new(bitwise_bus);

    GpuHarness {
        main: GpuTestChipHarness::with_capacity(
            main_executor,
            main_air,
            main_gpu_chip,
            main_chip,
            GPU_MAX_INS_CAPACITY,
        ),
        block_air: block_hasher_air,
        block_gpu: block_gpu_chip,
        block_cpu: block_hasher_chip,
        bitwise_air,
        bitwise_gpu,
    }
}

#[cfg(feature = "cuda")]
fn test_cuda_rand_sha2_multi_block<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness::<C>(&tester);

    let num_ops = 70;
    for _ in 1..=num_ops {
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.main.executor,
            &mut harness.main.dense_arena,
            &mut rng,
            C::OPCODE,
            None,
            None,
        );
    }

    harness
        .main
        .dense_arena
        .get_record_seeker::<Sha2RecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.main.matrix_arena);

    let mut tester = tester.build();
    tester = tester.load_gpu_harness(harness.main);
    tester = tester.load_and_compare(
        harness.block_air,
        harness.block_gpu,
        (),
        harness.block_cpu,
        (),
    );
    tester = tester.load_periphery(harness.bitwise_air, harness.bitwise_gpu);
    tester.finalize().simple_test().unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_sha256_multi_block() {
    test_cuda_rand_sha2_multi_block::<Sha256Config>();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_sha512_multi_block() {
    test_cuda_rand_sha2_multi_block::<Sha512Config>();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_sha384_multi_block() {
    test_cuda_rand_sha2_multi_block::<Sha384Config>();
}

#[cfg(feature = "cuda")]
fn test_cuda_sha2_known_vectors<C: Sha2Config + 'static>(test_vectors: &[(&str, &str)]) {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness::<C>(&tester);

    for (input, expected_hex) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected = Vec::from_hex(expected_hex).unwrap();
        // Sanity-check the expected digest matches the config’s hash.
        assert_eq!(C::hash(&input).as_slice(), expected.as_slice());
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.main.executor,
            &mut harness.main.dense_arena,
            &mut rng,
            C::OPCODE,
            Some(&input),
            Some(input.len()),
        );
    }
    // No block-hasher arena needed; GPU block chip ignores the arena input.
    harness
        .main
        .dense_arena
        .get_record_seeker::<Sha2RecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.main.matrix_arena);

    let mut tester = tester.build();
    tester = tester.load_gpu_harness(harness.main);
    tester = tester.load_and_compare(
        harness.block_air,
        harness.block_gpu,
        (),
        harness.block_cpu,
        (),
    );
    tester = tester.load_periphery(harness.bitwise_air, harness.bitwise_gpu);
    tester.finalize().simple_test().unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha256_known_vectors() {
    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05",
        ),
        ("5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c", "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7"),
        ("9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e", "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23")
    ];
    test_cuda_sha2_known_vectors::<Sha256Config>(&test_vectors);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha512_known_vectors() {
    let test_vectors = [
        (
            "",
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        ),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "eb576959c531f116842c0cc915a29c8f71d7a285c894c349b83469002ef093d51f9f14ce4248488bff143025e47ed27c12badb9cd43779cb147408eea062d583"
        ),
        (
            "9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e",
            "8d215ee6dc26757c210db0dd00c1c6ed16cc34dbd4bb0fa10c1edb6b62d5ab16aea88c881001b173d270676daf2d6381b5eab8711fa2f5589c477c1d4b84774f"
        ),
    ];
    test_cuda_sha2_known_vectors::<Sha512Config>(&test_vectors);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha384_known_vectors() {
    let test_vectors = [
        (
            "",
            "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b",
        ),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "63e3061aab01f335ea3a4e617b9d14af9b63a5240229164ee962f6d5335ff25f0f0bf8e46723e83c41b9d17413b6a3c7",
        ),
        (
            "9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e",
            "904a90010d772a904a35572fdd4bdf1dd253742e47872c8a18e2255f66fa889e44781e65487a043f435daa53c496a53e",
        ),
    ];
    test_cuda_sha2_known_vectors::<Sha384Config>(&test_vectors);
}

// GPU edge-case length tests mirroring the CPU suite.
#[cfg(feature = "cuda")]
fn cuda_sha2_edge_test_lengths<C: Sha2Config + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness::<C>(&tester);

    // check every possible input length modulo block size
    for i in (C::BLOCK_BYTES + 1)..=(2 * C::BLOCK_BYTES) {
        set_and_execute_full_message::<_, C, _>(
            &mut tester,
            &mut harness.main.executor,
            &mut harness.main.dense_arena,
            &mut rng,
            C::OPCODE,
            None,
            Some(i),
        );
    }

    harness
        .main
        .dense_arena
        .get_record_seeker::<Sha2RecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.main.matrix_arena);

    let mut tester = tester.build();
    tester = tester.load_gpu_harness(harness.main);
    tester = tester.load_and_compare(
        harness.block_air,
        harness.block_gpu,
        (),
        harness.block_cpu,
        (),
    );
    tester = tester.load_periphery(harness.bitwise_air, harness.bitwise_gpu);
    tester.finalize().simple_test().unwrap();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha256_edge_test_lengths() {
    cuda_sha2_edge_test_lengths::<Sha256Config>();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha512_edge_test_lengths() {
    cuda_sha2_edge_test_lengths::<Sha512Config>();
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha384_edge_test_lengths() {
    cuda_sha2_edge_test_lengths::<Sha384Config>();
}
