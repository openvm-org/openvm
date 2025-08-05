use std::{array, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, DenseRecordArena, MatrixRecordArena, PreflightExecutor,
    },
    utils::get_random_message,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_sha256_transpiler::Rv32Sha256Opcode::{self, *};
use openvm_stark_backend::{interaction::BusIndex, p3_field::FieldAlgebra};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{Sha256VmAir, Sha256VmChip, Sha256VmStep};
use crate::{
    sha256_chip::trace::Sha256VmRecordLayout, sha256_solve, Sha256VmDigestCols, Sha256VmFiller,
    Sha256VmRoundCols,
};

type F = BabyBear;
const SELF_BUS_IDX: BusIndex = 28;
const MAX_INS_CAPACITY: usize = 4096;
type Harness<RA> = TestChipHarness<F, Sha256VmStep, Sha256VmAir, Sha256VmChip<F>, RA>;

fn create_test_chips<RA: Arena>(
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

    let air = Sha256VmAir::new(
        tester.system_port(),
        bitwise_chip.bus(),
        tester.address_bits(),
        SELF_BUS_IDX,
    );
    let executor = Sha256VmStep::new(Rv32Sha256Opcode::CLASS_OFFSET, tester.address_bits());
    let chip = Sha256VmChip::new(
        Sha256VmFiller::new(bitwise_chip.clone(), tester.address_bits()),
        tester.memory_helper(),
    );

    let harness = Harness::<RA>::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena>(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness<RA>,
    rng: &mut StdRng,
    opcode: Rv32Sha256Opcode,
    message: Option<&[u8]>,
    len: Option<usize>,
) where
    Sha256VmStep: PreflightExecutor<F, RA>,
{
    let len = len.unwrap_or(rng.gen_range(1..3000));
    let tmp = get_random_message(rng, len);
    let message: &[u8] = message.unwrap_or(&tmp);
    let len = message.len();

    let rd = gen_pointer(rng, 4);
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);

    let max_mem_ptr: u32 = 1 << tester.address_bits();
    let dst_ptr = rng.gen_range(0..max_mem_ptr);
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
        harness,
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
    let (mut harness, bitwise) = create_test_chips(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, SHA256, None, None);
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
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
    let (mut harness, _) = create_test_chips::<MatrixRecordArena<F>>(&mut tester);

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
        set_and_execute(&mut tester, &mut harness, &mut rng, SHA256, None, None);
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

///////////////////////////////////////////////////////////////////////////////////////
/// DENSE TESTS
///
/// Ensure that the chip works as expected with dense records.
/// We first execute some instructions with a [DenseRecordArena] and transfer the records
/// to a [MatrixRecordArena]. After transferring we generate the trace and make sure that
/// all the constraints pass.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn dense_record_arena_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut sparse_harness, bitwise) = create_test_chips(&mut tester);

    {
        let mut dense_harness = create_test_chips::<DenseRecordArena>(&mut tester).0;

        let num_ops: usize = 10;
        for _ in 0..num_ops {
            set_and_execute(
                &mut tester,
                &mut dense_harness,
                &mut rng,
                SHA256,
                None,
                None,
            );
        }

        let mut record_interpreter = dense_harness
            .arena
            .get_record_seeker::<_, Sha256VmRecordLayout>();
        record_interpreter.transfer_to_matrix_arena(&mut sparse_harness.arena);
    }

    let tester = tester
        .build()
        .load(sparse_harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
