use openvm_circuit::arch::{
    testing::{memory::gen_pointer, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    SystemPort,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_sha256_transpiler::Rv32Sha2Opcode::{self, *};
use openvm_sha_air::{get_random_message, Sha256Config, Sha384Config, Sha512Config};
use openvm_stark_backend::{interaction::BusIndex, p3_field::FieldAlgebra};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{Sha2VmChip, ShaChipConfig};
use crate::{sha2_solve, ShaVmDigestColsRef, ShaVmRoundColsRef};

type F = BabyBear;
const BUS_IDX: BusIndex = 28;
fn set_and_execute<C: ShaChipConfig>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Sha2VmChip<F, C>,
    rng: &mut StdRng,
    opcode: Rv32Sha2Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) {
    let len = len.unwrap_or(rng.gen_range(1..100000));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let max_mem_ptr: u32 = 1
        << tester
            .memory_controller()
            .borrow()
            .mem_config()
            .pointer_max_bits;
    let dst_ptr = rng.gen_range(0..max_mem_ptr - C::DIGEST_SIZE as u32);
    let dst_ptr = dst_ptr ^ (dst_ptr & 3);
    tester.write(1, rd, dst_ptr.to_le_bytes().map(F::from_canonical_u8));
    let src_ptr = rng.gen_range(0..(max_mem_ptr - len as u32));
    let src_ptr = src_ptr ^ (src_ptr & 3);
    tester.write(1, rs1, src_ptr.to_le_bytes().map(F::from_canonical_u8));
    tester.write(1, rs2, len.to_le_bytes().map(F::from_canonical_u8));

    for (i, &byte) in message.iter().enumerate() {
        tester.write(2, src_ptr as usize + i, [F::from_canonical_u8(byte)]);
    }

    tester.execute(
        chip,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 2]),
    );

    let output = sha2_solve::<C>(message);
    if C::OPCODE_NAME == "SHA256" {
        assert_eq!(
            output
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>(),
            tester.read::<{ Sha256Config::DIGEST_SIZE }>(2, dst_ptr as usize)
        );
    } else if C::OPCODE_NAME == "SHA512" {
        // TODO: break into two reads
        assert_eq!(
            output
                .into_iter()
                .map(F::from_canonical_u8)
                .collect::<Vec<_>>(),
            tester.read::<{ Sha512Config::DIGEST_SIZE }>(2, dst_ptr as usize)
        );
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
fn rand_sha_test<C: ShaChipConfig + 'static>(opcode: Rv32Sha2Opcode) {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Sha2VmChip::<F, C>::new(
        SystemPort {
            execution_bus: tester.execution_bus(),
            program_bus: tester.program_bus(),
            memory_bridge: tester.memory_bridge(),
        },
        tester.address_bits(),
        bitwise_chip.clone(),
        BUS_IDX,
        Rv32Sha2Opcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

    let num_tests: usize = 3;
    for _ in 0..num_tests {
        set_and_execute::<C>(&mut tester, &mut chip, &mut rng, opcode, None, None);
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha_test::<Sha256Config>(SHA256);
}

#[test]
fn rand_sha512_test() {
    rand_sha_test::<Sha512Config>(SHA512);
}

#[test]
fn rand_sha384_test() {
    rand_sha_test::<Sha384Config>(SHA384);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
fn execute_roundtrip_sanity_test<C: ShaChipConfig>(opcode: Rv32Sha2Opcode) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Sha2VmChip::<F, C>::new(
        SystemPort {
            execution_bus: tester.execution_bus(),
            program_bus: tester.program_bus(),
            memory_bridge: tester.memory_bridge(),
        },
        tester.address_bits(),
        bitwise_chip.clone(),
        BUS_IDX,
        Rv32Sha2Opcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

    println!(
        "ShaVmDigestColsRef::<F>::width::<C>(): {}",
        ShaVmDigestColsRef::<F>::width::<C>()
    );
    println!(
        "ShaVmRoundColsRef::<F>::width::<C>(): {}",
        ShaVmRoundColsRef::<F>::width::<C>()
    );

    let num_tests: usize = 1;
    for _ in 0..num_tests {
        set_and_execute::<C>(&mut tester, &mut chip, &mut rng, opcode, None, None);
    }
}

#[test]
fn sha256_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha256Config>(SHA256);
}

#[test]
fn sha512_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha512Config>(SHA512);
}

#[test]
fn sha384_roundtrip_sanity_test() {
    execute_roundtrip_sanity_test::<Sha384Config>(SHA384);
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
