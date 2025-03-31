use openvm_circuit::arch::{
    testing::{memory::gen_pointer, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    SystemPort,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_sha256_air::get_random_message;
use openvm_sha256_transpiler::Rv32Sha256Opcode::{self, *};
use openvm_stark_backend::{interaction::BusIndex, p3_field::FieldAlgebra};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::Sha256VmChip;
use crate::{sha256_solve, Sha256VmDigestCols, Sha256VmRoundCols};

type F = BabyBear;
const BUS_IDX: BusIndex = 28;
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Sha256VmChip<F>,
    rng: &mut StdRng,
    opcode: Rv32Sha256Opcode,
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
    let dst_ptr = rng.gen_range(0..max_mem_ptr);
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

    let output = sha256_solve(message);
    assert_eq!(
        output.map(F::from_canonical_u8),
        tester.read::<32>(2, dst_ptr as usize)
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
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Sha256VmChip::new(
        SystemPort {
            execution_bus: tester.execution_bus(),
            program_bus: tester.program_bus(),
            memory_bridge: tester.memory_bridge(),
        },
        tester.address_bits(),
        bitwise_chip.clone(),
        BUS_IDX,
        Rv32Sha256Opcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

    let num_tests: usize = 3;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, SHA256, None, None);
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
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
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let mut chip = Sha256VmChip::new(
        SystemPort {
            execution_bus: tester.execution_bus(),
            program_bus: tester.program_bus(),
            memory_bridge: tester.memory_bridge(),
        },
        tester.address_bits(),
        bitwise_chip.clone(),
        BUS_IDX,
        Rv32Sha256Opcode::CLASS_OFFSET,
        tester.offline_memory_mutex_arc(),
    );

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
        set_and_execute(&mut tester, &mut chip, &mut rng, SHA256, None, None);
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
