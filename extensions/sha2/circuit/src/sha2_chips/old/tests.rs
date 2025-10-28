use std::array;

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        DenseRecordArena, InsExecutorE1, InstructionExecutor, NewVmChipWrapper,
    },
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_sha2_air::{Sha256Config, Sha2Variant, Sha384Config, Sha512Config};
use openvm_sha2_transpiler::Rv32Sha2Opcode;
use openvm_stark_backend::{interaction::BusIndex, p3_field::FieldAlgebra};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{Sha2BlockHasheAir, Sha2ChipConfig, Sha2VmChip, Sha2VmStep};
use crate::{
    sha2_chip::trace::Sha2VmRecordLayout, sha2_solve, Sha2BlockHasherDigestColsRef,
    Sha2BlockHasherRoundColsRef,
};

type F = BabyBear;
const SELF_BUS_IDX: BusIndex = 28;
const MAX_INS_CAPACITY: usize = 8192;
type Sha2VmChipDense<C> =
    NewVmChipWrapper<F, Sha2BlockHasheAir<C>, Sha2VmStep<C>, DenseRecordArena>;

fn create_test_chips<C: Sha2ChipConfig>(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Sha2VmChip<F, C>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Sha2VmChip::<F, C>::new(
        Sha2BlockHasheAir::new(
            tester.system_port(),
            bitwise_bus,
            tester.address_bits(),
            SELF_BUS_IDX,
        ),
        Sha2VmStep::new(
            bitwise_chip.clone(),
            Rv32Sha2Opcode::CLASS_OFFSET,
            tester.address_bits(),
        ),
        tester.memory_helper(),
    );
    chip.set_trace_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

fn set_and_execute<E: InstructionExecutor<F>, C: Sha2ChipConfig>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: Rv32Sha2Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let len = len.unwrap_or(rng.gen_range(1..3000));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let max_mem_ptr: u32 = 1 << tester.address_bits();
    let dst_ptr = rng.gen_range(0..max_mem_ptr - C::DIGEST_SIZE as u32);
    let dst_ptr = dst_ptr ^ (dst_ptr & 3);
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    let src_ptr = rng.gen_range(0..(max_mem_ptr - len as u32));
    let src_ptr = src_ptr ^ (src_ptr & 3);
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    message.chunks(4).enumerate().for_each(|(i, chunk)| {
        let chunk: [&u8; 4] = array::from_fn(|i| chunk.get(i).unwrap_or(&0));
        tester.write(
            2,
            src_ptr as usize + i * 4,
            chunk.map(|&x| F::from_canonical_u8(x)),
        );
    });

    tester.execute(
        chip,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let output = sha2_solve::<C>(message);
    match C::VARIANT {
        Sha2Variant::Sha256 => {
            assert_eq!(
                output
                    .into_iter()
                    .map(F::from_canonical_u8)
                    .collect::<Vec<_>>(),
                tester.read::<{ Sha256Config::DIGEST_SIZE }>(2, dst_ptr as usize)
            );
        }
        Sha2Variant::Sha512 | Sha2Variant::Sha384 => {
            let mut output = output;
            output.extend(std::iter::repeat(0u8).take(C::HASH_SIZE));
            let output = output
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>();
            for i in 0..C::NUM_WRITES {
                assert_eq!(
                    output[i * C::WRITE_SIZE..(i + 1) * C::WRITE_SIZE],
                    tester.read::<{ Sha512Config::WRITE_SIZE }>(
                        2,
                        dst_ptr as usize + i * C::WRITE_SIZE
                    )
                );
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
fn rand_sha_test<C: Sha2ChipConfig + 'static>() {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chips::<C>(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute::<_, C>(&mut tester, &mut chip, &mut rng, C::OPCODE, None, None);
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha_test::<Sha256Config>();
}

#[test]
fn rand_sha512_test() {
    rand_sha_test::<Sha512Config>();
}

#[test]
fn rand_sha384_test() {
    rand_sha_test::<Sha384Config>();
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
fn execute_roundtrip_sanity_test<C: Sha2ChipConfig>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, _) = create_test_chips::<C>(&mut tester);

    println!(
        "Sha2VmDigestColsRef::<u8>::width::<C>(): {}",
        Sha2BlockHasherDigestColsRef::<u8>::width::<C>()
    );
    println!(
        "Sha2VmRoundColsRef::<u8>::width::<C>(): {}",
        Sha2BlockHasherRoundColsRef::<u8>::width::<C>()
    );

    let num_tests: usize = 1;
    for _ in 0..num_tests {
        set_and_execute::<_, C>(&mut tester, &mut chip, &mut rng, C::OPCODE, None, None);
    }
}

#[test]
fn sha256_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha256Config>();
}

#[test]
fn sha512_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha512Config>();
}

#[test]
fn sha384_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha384Config>();
}

#[test]
fn sha256_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = sha2_solve::<Sha256Config>(input);
    let expected: [u8; 32] = [
        99, 196, 61, 185, 226, 212, 131, 80, 154, 248, 97, 108, 157, 55, 200, 226, 160, 73, 207,
        46, 245, 169, 94, 255, 42, 136, 193, 15, 40, 133, 173, 22,
    ];
    assert_eq!(output, expected);
}

#[test]
fn sha512_solve_sanity_check() {
    let input = b"Axiom is the best! Axiom is the best! Axiom is the best! Axiom is the best!";
    let output = sha2_solve::<Sha512Config>(input);
    // verified manually against the sha512 command line tool
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
    let output = sha2_solve::<Sha384Config>(input);
    let expected: [u8; 48] = [
        134, 227, 167, 229, 35, 110, 115, 174, 10, 27, 197, 116, 56, 144, 150, 36, 152, 190, 212,
        120, 26, 243, 125, 4, 2, 60, 164, 195, 218, 219, 255, 143, 240, 75, 158, 126, 102, 105, 8,
        202, 142, 240, 230, 161, 162, 152, 111, 71,
    ];
    assert_eq!(output, expected);
}

///////////////////////////////////////////////////////////////////////////////////////
/// DENSE TESTS
///
/// Ensure that the chip works as expected with dense records.
/// We first execute some instructions with a [DenseRecordArena] and transfer the records
/// to a [MatrixRecordArena]. After transferring we generate the trace and make sure that
/// all the constraints pass.
///////////////////////////////////////////////////////////////////////////////////////
fn create_test_chip_dense<C: Sha2ChipConfig>(
    tester: &mut VmChipTestBuilder<F>,
) -> Sha2VmChipDense<C> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Sha2VmChipDense::<C>::new(
        Sha2BlockHasheAir::<C>::new(
            tester.system_port(),
            bitwise_chip.bus(),
            tester.address_bits(),
            SELF_BUS_IDX,
        ),
        Sha2VmStep::<C>::new(
            bitwise_chip.clone(),
            Rv32Sha2Opcode::CLASS_OFFSET,
            tester.address_bits(),
        ),
        tester.memory_helper(),
    );

    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

fn dense_record_arena_test<C: Sha2ChipConfig + 'static>() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut sparse_chip, bitwise_chip) = create_test_chips::<C>(&mut tester);

    {
        let mut dense_chip = create_test_chip_dense::<C>(&mut tester);

        let num_ops: usize = 10;
        for _ in 0..num_ops {
            set_and_execute::<_, C>(
                &mut tester,
                &mut dense_chip,
                &mut rng,
                C::OPCODE,
                None,
                None,
            );
        }

        let mut record_interpreter = dense_chip
            .arena
            .get_record_seeker::<_, Sha2VmRecordLayout<C>>();
        record_interpreter.transfer_to_matrix_arena(&mut sparse_chip.arena);
    }

    let tester = tester
        .build()
        .load(sparse_chip)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn sha256_dense_record_arena_test() {
    dense_record_arena_test::<Sha256Config>();
}

#[test]
fn sha512_dense_record_arena_test() {
    dense_record_arena_test::<Sha512Config>();
}

#[test]
fn sha384_dense_record_arena_test() {
    dense_record_arena_test::<Sha384Config>();
}
