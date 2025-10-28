use std::{array, borrow::BorrowMut, sync::Arc};

use hex::FromHex;
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
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS},
    LocalOpcode,
};
use openvm_sha2_transpiler::Rv32Sha2Opcode::{self, *};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::FieldAlgebra,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{trace::Sha2BlockHasherRecordMut, Sha2BlockHasherChipGpu},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use crate::{
    Sha2BlockHasherAir, Sha2BlockHasherChip, Sha2BlockHasherConfig, Sha2BlockHasherFiller,
    Sha2Config, Sha2VmExecutor,
};

const SHA2_BUS_IDX: BusIndex = 28;
type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA, C> =
    TestChipHarness<F, Sha2VmExecutor<C>, Sha2BlockHasherAir<C>, Sha2BlockHasherChip<F>, RA>;

fn create_harness_fields<C: Sha2Config>(
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
    bitwise_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (
    Sha2BlockHasherAir<C>,
    Sha2VmExecutor<C>,
    Sha2BlockHasherChip<F>,
) {
    let air = Sha2BlockHasherAir::new(bitwise_chip.bus(), SHA2_BUS_IDX);
    let executor = Sha2VmExecutor::<C>::new(Rv32Sha2Opcode::CLASS_OFFSET, address_bits);
    let chip = Sha2BlockHasherChip::new(
        Sha2BlockHasherFiller::new(bitwise_chip, address_bits),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_harness<RA: Arena, C: Sha2Config>(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness<RA, C>,
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

    let harness = Harness::<RA, C>::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

// taken from keccak256 tests
// WIP
#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, C: Sha2Config, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: Rv32Sha2Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
    expected_output: Option<[u8; 32]>,
) {
    let len = len.unwrap_or(rng.gen_range(1..3000));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let dst_ptr = gen_pointer(rng, 4);
    let src_ptr = gen_pointer(rng, 4);
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    message.chunks(4).enumerate().for_each(|(i, chunk)| {
        let rng = rng.gen();
        let chunk: [&u8; 4] = array::from_fn(|i| chunk.get(i).unwrap_or(&rng));
        tester.write(
            RV32_MEMORY_AS as usize,
            src_ptr + i * 4,
            chunk.map(|&x| F::from_canonical_u8(x)),
        );
    });

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let expected_output = expected_output.unwrap_or(keccak256(message));
    println!("expected_output: {:?}", expected_output);
    println!("keccak256(message): {:?}", keccak256(message));
    assert_eq!(
        expected_output.map(F::from_canonical_u8),
        tester.read(RV32_MEMORY_AS as usize, dst_ptr)
    );
}

// old test code

fn create_chip_with_rand_records<C: Sha2BlockHasherConfig + 'static>(
) -> (Sha2TestChip<C>, SharedBitwiseOperationLookupChip<8>) {
    let mut rng = create_seeded_rng();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|i| {
            (
                (0..C::BLOCK_U8S)
                    .map(|_| rng.gen::<u8>())
                    .collect::<Vec<_>>(),
                rng.gen::<bool>() || i == len - 1,
            )
        })
        .collect();
    let chip = Sha2TestChip {
        air: Sha2TestAir {
            sub_air: Sha2BlockHasherAir::<C>::new(bitwise_bus, SELF_BUS_IDX),
        },
        step: Sha2StepHelper::<C>::new(),
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    (chip, bitwise_chip)
}

fn rand_sha2_test<C: Sha2BlockHasherConfig + 'static>() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records::<C>();
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha2_test::<Sha256Config>();
}

#[test]
fn rand_sha512_test() {
    rand_sha2_test::<Sha512Config>();
}

#[test]
fn rand_sha384_test() {
    rand_sha2_test::<Sha384Config>();
}

fn negative_sha2_test_bad_final_hash<C: Sha2BlockHasherConfig + 'static>() {
    let tester = VmChipTestBuilder::default();
    let (chip, bitwise_chip) = create_chip_with_rand_records::<C>();

    // Set the final_hash to all zeros
    let modify_trace = |trace: &mut RowMajorMatrix<F>| {
        trace.row_chunks_exact_mut(1).for_each(|row| {
            let mut row_slice = row.row_slice(0).to_vec();
            let mut cols: Sha2DigestColsRefMut<F> =
                Sha2DigestColsRefMut::from::<C>(&mut row_slice[..C::DIGEST_WIDTH]);
            if cols.flags.is_last_block.is_one() && cols.flags.is_digest_row.is_one() {
                for i in 0..C::HASH_WORDS {
                    for j in 0..C::WORD_U8S {
                        cols.final_hash[[i, j]] = F::ZERO;
                    }
                }
                row.values.copy_from_slice(&row_slice);
            }
        });
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(VerificationError::OodEvaluationMismatch);
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
