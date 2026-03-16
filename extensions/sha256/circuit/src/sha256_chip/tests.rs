use std::{array, borrow::BorrowMut, sync::Arc};

use hex::FromHex;
use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, MatrixRecordArena, PreflightExecutor,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_sha256_air::{
    get_sha256_num_blocks, Sha256FillerHelper, SHA256_BLOCK_U8S, SHA256_H, SHA256_ROWS_PER_BLOCK,
};
use openvm_sha256_transpiler::Rv32Sha256Opcode::{self, *};
use openvm_stark_backend::{
    interaction::BusIndex,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::DenseMatrix, Matrix},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{Sha256VmChipGpu, Sha256VmRecordMut},
    openvm_circuit::arch::testing::{
        default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness,
    },
};

use super::{Sha256VmAir, Sha256VmChip, Sha256VmExecutor};
use crate::{
    sha256_solve, Sha256VmDigestCols, Sha256VmFiller, Sha256VmRoundCols, SHA256VM_CONTROL_WIDTH,
    SHA256VM_DIGEST_WIDTH, SHA256VM_WIDTH,
};

type F = BabyBear;
const SELF_BUS_IDX: BusIndex = 28;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> = TestChipHarness<F, Sha256VmExecutor, Sha256VmAir, Sha256VmChip<F>, RA>;

fn create_harness_fields(
    system_port: SystemPort,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
    address_bits: usize,
) -> (Sha256VmAir, Sha256VmExecutor, Sha256VmChip<F>) {
    let air = Sha256VmAir::new(system_port, bitwise_chip.bus(), address_bits, SELF_BUS_IDX);
    let executor = Sha256VmExecutor::new(Rv32Sha256Opcode::CLASS_OFFSET, address_bits);
    let chip = Sha256VmChip::new(
        Sha256VmFiller::new(bitwise_chip, address_bits),
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
        tester.system_port(),
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
    opcode: Rv32Sha256Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let len = len.unwrap_or(rng.random_range(1..3000));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let dst_ptr = gen_pointer(rng, 4);
    let src_ptr = gen_pointer(rng, 4);
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_u8));
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_u8));

    // Adding random memory after the message
    let num_blocks = get_sha256_num_blocks(len as u32) as usize;
    for offset in (0..num_blocks * SHA256_BLOCK_U8S).step_by(4) {
        let chunk: [F; 4] = array::from_fn(|i| {
            if offset + i < message.len() {
                F::from_u8(message[offset + i])
            } else {
                F::from_u8(rng.random())
            }
        });

        tester.write(RV32_MEMORY_AS as usize, src_ptr + offset, chunk);
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let output = sha256_solve(message);
    assert_eq!(
        output.map(F::from_u8),
        tester.read::<32>(RV32_MEMORY_AS as usize, dst_ptr)
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_sha256_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            SHA256,
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

#[test]
fn sha256_edge_test_lengths() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&mut tester);

    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05",
        ),
        ("5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c", "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7"),
        ("9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e", "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23")
    ];

    for (input, _) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            SHA256,
            Some(&input),
            None,
        );
    }

    // check every possible input length modulo 64
    for i in 65..=128 {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            SHA256,
            None,
            Some(i),
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

/// Execute sha256 with len=16, then mutate `len_data` on the last digest row
/// to the byte decomposition of `16 + p`.
///
/// After pranking the digest row, regenerate the inner SHA block trace and the
/// cross-row helper cells so the trace stays consistent under the digest-row /
/// round-row overlap in the subair. Also compensate the memory bus so the
/// pranked register read is accepted.
///
/// The range check on `len_data[3]` rejects this malicious trace, even after
/// regenerating the overlapped SHA helper cells and compensating the memory
/// bus interactions.
#[test]
fn len_data_range_exploit_is_rejected_with_check() {
    use std::cell::Cell;

    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, (bitwise_air, bitwise_chip)) =
        create_harness::<MatrixRecordArena<F>>(&mut tester);

    let message = [0xab_u8; 16];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        SHA256,
        Some(&message),
        None,
    );

    // BabyBear modulus p = 2013265921. 16 + p = 2013265937 = 0x78000011.
    // LE bytes: [0x11, 0x00, 0x00, 0x78] = [17, 0, 0, 120].
    // compose([17,0,0,120]) = 17 + 120 * 2^24 = 2013265937 ≡ 16 (mod p).
    let pranked_len: u32 = 16u32.wrapping_add(2013265921);
    let pranked_bytes = pranked_len.to_le_bytes();
    let address_bits = tester.address_bits();
    let sha_helper = Sha256FillerHelper::new();
    let padded_block_words: [u32; 16] = {
        let mut padded_block = [0u8; SHA256_BLOCK_U8S];
        padded_block[..message.len()].copy_from_slice(&message);
        padded_block[message.len()] = 1 << (RV32_CELL_BITS - 1);
        padded_block[SHA256_BLOCK_U8S - 4..]
            .copy_from_slice(&((message.len() as u32) << 3).to_be_bytes());
        array::from_fn(|i| u32::from_be_bytes(padded_block[i * 4..(i + 1) * 4].try_into().unwrap()))
    };

    // Capture values from the digest row needed for memory bus compensation.
    let orig_len_data_cell: Cell<[F; 4]> = Cell::new([F::ZERO; 4]);
    let rs2_ptr_cell: Cell<u32> = Cell::new(0);
    let dst_ptr_cell: Cell<u32> = Cell::new(0);
    let src_ptr_cell: Cell<u32> = Cell::new(0);
    let prev_ts_cell: Cell<u32> = Cell::new(0);
    let read_ts_cell: Cell<u32> = Cell::new(0);

    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let width = trace.width();
        let mut pranked_digest = false;
        for row_idx in 0..trace.height() {
            let row = &mut trace.values[row_idx * width..(row_idx + 1) * width];
            let digest: &mut Sha256VmDigestCols<F> = row[..SHA256VM_DIGEST_WIDTH].borrow_mut();
            if digest.inner.flags.is_digest_row.is_one()
                && digest.inner.flags.is_last_block.is_one()
            {
                // Extract values for memory bus compensation before pranking.
                orig_len_data_cell.set(digest.len_data);
                rs2_ptr_cell.set(digest.rs2_ptr.as_canonical_u32());
                dst_ptr_cell.set(u32::from_le_bytes(
                    digest.dst_ptr.map(|x| x.as_canonical_u32() as u8),
                ));
                src_ptr_cell.set(u32::from_le_bytes(
                    digest.src_ptr.map(|x| x.as_canonical_u32() as u8),
                ));
                let from_ts = digest.from_state.timestamp.as_canonical_u32();
                // rs2 is the 3rd register read, so its timestamp is from_state.timestamp + 2.
                read_ts_cell.set(from_ts + 2);
                prev_ts_cell.set(
                    digest.register_reads_aux[2]
                        .get_base()
                        .prev_timestamp
                        .as_canonical_u32(),
                );

                digest.len_data = pranked_bytes.map(F::from_u8);
                pranked_digest = true;
            }
        }

        assert!(
            pranked_digest,
            "failed to find the final digest row to prank"
        );

        // Recompute the inner SHA trace and helper rows so the digest-row /
        // round-row overlap is consistent with the pranked len_data.
        bitwise_chip.clear();
        sha_helper.generate_block_trace(
            &mut trace.values[..SHA256VM_WIDTH * SHA256_ROWS_PER_BLOCK],
            SHA256VM_WIDTH,
            SHA256VM_CONTROL_WIDTH,
            &padded_block_words,
            bitwise_chip.as_ref(),
            &SHA256_H,
            true,
            1,
            0,
        );
        sha_helper.generate_missing_cells(
            &mut trace.values[SHA256VM_WIDTH..SHA256VM_WIDTH * (SHA256_ROWS_PER_BLOCK + 1)],
            SHA256VM_WIDTH,
            SHA256VM_CONTROL_WIDTH,
        );

        // Re-add the VM-layer pointer range check query for the last digest row.
        let msl_rshift = ((RV32_REGISTER_NUM_LIMBS - 1) * RV32_CELL_BITS) as u32;
        let msl_lshift = (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS - address_bits) as u32;
        bitwise_chip.request_range(
            (dst_ptr_cell.get() >> msl_rshift) << msl_lshift,
            (src_ptr_cell.get() >> msl_rshift) << msl_lshift,
        );
    };

    disable_debug_builder();
    let mut tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery((bitwise_air, bitwise_chip));

    // Compensate memory bus: the SHA256 trace now claims to read pranked_bytes from
    // the rs2 register, but the memory controller trace still reflects the original
    // [16,0,0,0]. We add compensating messages to MemoryDummyChip so the trace is
    // otherwise valid; the remaining rejection path should be the len_data range
    // check itself.
    let orig_data = orig_len_data_cell.get();
    let pranked_data: [F; 4] = pranked_bytes.map(F::from_u8);
    let rs2_ptr = rs2_ptr_cell.get();
    let prev_ts = prev_ts_cell.get();
    let read_ts = read_ts_cell.get();

    let memory = tester.memory.as_mut().unwrap();
    let chip4 = memory.chip_for_block.get_mut(&4).unwrap();

    chip4.receive(1, rs2_ptr, &orig_data, prev_ts);
    chip4.send(1, rs2_ptr, &orig_data, read_ts);
    chip4.send(1, rs2_ptr, &pranked_data, prev_ts);
    chip4.receive(1, rs2_ptr, &pranked_data, read_ts);

    let tester = tester.finalize();
    let result = tester.simple_test();
    if let Err(ref e) = result {
        eprintln!("DEBUG: verification error = {:?}", e);
    }
    assert_eq!(
        result.err(),
        Some(VerificationError::ChallengePhaseError),
        "Trace with len_data decomposing to 16+p ({}) should be rejected by the len_data range check.",
        pranked_len,
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
    let (mut harness, _) = create_harness::<MatrixRecordArena<F>>(&mut tester);

    println!(
        "Sha256VmDigestCols::width(): {}",
        Sha256VmDigestCols::<F>::width()
    );
    println!(
        "Sha256VmRoundCols::width(): {}",
        Sha256VmRoundCols::<F>::width()
    );
    let num_tests: usize = 1;
    for _ in 0..num_tests {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            SHA256,
            None,
            None,
        );
    }
}

#[test]
fn sha256_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = sha256_solve(input);
    let expected: [u8; 32] = [
        99, 196, 61, 185, 226, 212, 131, 80, 154, 248, 97, 108, 157, 55, 200, 226, 160, 73, 207,
        46, 245, 169, 94, 255, 42, 136, 193, 15, 40, 133, 173, 22,
    ];
    assert_eq!(output, expected);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Sha256VmExecutor, Sha256VmAir, Sha256VmChipGpu, Sha256VmChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    const GPU_MAX_INS_CAPACITY: usize = 8192;

    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.system_port(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
        tester.address_bits(),
    );
    let gpu_chip = Sha256VmChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.address_bits() as u32,
        tester.timestamp_max_bits() as u32,
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, GPU_MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_sha256_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    let num_ops = 70;
    for i in 1..=num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            SHA256,
            None,
            Some(i),
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<Sha256VmRecordMut, _>()
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
fn test_cuda_sha256_known_vectors() {
    let mut rng = create_seeded_rng();
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());

    let mut harness = create_cuda_harness(&tester);

    let test_vectors = [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (
            "98c1c0bdb7d5fea9a88859f06c6c439f",
            "b6b2c9c9b6f30e5c66c977f1bd7ad97071bee739524aecf793384890619f2b05",
        ),
        ("5b58f4163e248467cc1cd3eecafe749e8e2baaf82c0f63af06df0526347d7a11327463c115210a46b6740244eddf370be89c", "ac0e25049870b91d78ef6807bb87fce4603c81abd3c097fba2403fd18b6ce0b7"),
        ("9ad198539e3160194f38ac076a782bd5210a007560d1fce9ef78f8a4a5e4d78c6b96c250cff3520009036e9c6087d5dab587394edda862862013de49a12072485a6c01165ec0f28ffddf1873fbd53e47fcd02fb6a5ccc9622d5588a92429c663ce298cb71b50022fc2ec4ba9f5bbd250974e1a607b165fee16e8f3f2be20d7348b91a2f518ce928491900d56d9f86970611580350cee08daea7717fe28a73b8dcfdea22a65ed9f5a09198de38e4e4f2cc05b0ba3dd787a5363ab6c9f39dcb66c1a29209b1d6b1152769395df8150b4316658ea6ab19af94903d643fcb0ae4d598035ebe73c8b1b687df1ab16504f633c929569c6d0e5fae6eea43838fbc8ce2c2b43161d0addc8ccf945a9c4e06294e56a67df0000f561f61b630b1983ba403e775aaeefa8d339f669d1e09ead7eae979383eda983321e1743e5404b4b328da656de79ff52d179833a6bd5129f49432d74d001996c37c68d9ab49fcff8061d193576f396c20e1f0d9ee83a51290ba60efa9c3cb2e15b756321a7ca668cdbf63f95ec33b1c450aa100101be059dc00077245b25a6a66698dee81953ed4a606944076e2858b1420de0095a7f60b08194d6d9a997009d345c71f63a7034b976e409af8a9a040ac7113664609a7adedb76b2fadf04b0348392a1650526eb2a4d6ed5e4bbcda8aabc8488b38f4f5d9a398103536bb8250ed82a9b9825f7703c263f9e", "080ad71239852124fc26758982090611b9b19abf22d22db3a57f67a06e984a23")
    ];

    for (input, _) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();

        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            SHA256,
            Some(&input),
            None,
        );
    }

    harness
        .dense_arena
        .get_record_seeker::<Sha256VmRecordMut, _>()
        .transfer_to_matrix_arena(&mut harness.matrix_arena);

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
