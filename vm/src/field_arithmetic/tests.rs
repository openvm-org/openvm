use core::borrow::Borrow;
use std::ops::Deref;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::Rng;

use super::{FieldArithmeticAir, FieldArithmeticChip};
use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::MachineChip,
        instructions::{OpCode::*, FIELD_ARITHMETIC_INSTRUCTIONS},
        testing::{ExecutionTester, MachineChipTester, MemoryTester},
    },
    cpu::trace::Instruction,
    field_arithmetic::columns::{FieldArithmeticCols, FieldArithmeticIoCols},
};

#[test]
fn field_arithmetic_air_test() {
    let num_ops = 19;
    let elem_range = || 1..=100;
    let xz_address_space_range = || 1usize..=2;
    let y_address_space_range = || 0usize..=2;
    let address_range = || 0usize..1 << 29;

    let execution_bus = ExecutionBus(0);
    let memory_bus = 1;
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(execution_bus, memory_tester.get_memory_chip());

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = FIELD_ARITHMETIC_INSTRUCTIONS[rng.gen_range(0..4)];

        let operand1 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));
        let operand2 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));

        let as_d = rng.gen_range(xz_address_space_range());
        let as_e = rng.gen_range(y_address_space_range());
        let address1 = rng.gen_range(address_range());
        let address2 = if as_e == 0 {
            operand2.as_canonical_u32() as usize
        } else {
            rng.gen_range(address_range())
        };
        let result_address = rng.gen_range(address_range());
        assert!(address1.abs_diff(address2) >= 4);
        println!(
            "d = {}, e = {}, result_addr = {}, addr1 = {}, addr2 = {}",
            as_d, as_e, result_address, address1, address2
        );

        let result = FieldArithmeticAir::solve(opcode, (operand1, operand2));

        memory_tester.install(as_d, address1, [operand1]);
        if as_e != 0 {
            memory_tester.install(as_e, address2, [operand2]);
        }
        execution_tester.execute(
            &mut field_arithmetic_chip,
            Instruction::from_usize(opcode, result_address, address1, address2, as_d, as_e),
        );
        memory_tester.expect(as_d, result_address, [result]);
        memory_tester.check();
    }

    // positive test
    MachineChipTester::default()
        .add(&mut execution_tester)
        .add(&mut memory_tester)
        .add(&mut field_arithmetic_chip)
        .simple_test()
        .expect("Verification failed");

    return;

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    // negative test pranking each IO value
    for height in 0..num_ops {
        let mut arithmetic_trace = field_arithmetic_chip.generate_trace();
        for width in 0..FieldArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            arithmetic_trace.row_mut(height)[width] = prank_value;
        }

        let test_result = MachineChipTester::default()
            .add(&mut execution_tester)
            .add(&mut memory_tester)
            .add_with_custom_trace(&mut field_arithmetic_chip, arithmetic_trace)
            .simple_test();

        // Run a test after pranking each row
        assert_eq!(
            test_result,
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        )
    }
}

#[test]
fn au_air_zero_div_zero() {
    let execution_bus = ExecutionBus(0);
    let memory_bus = 1;
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(execution_bus, memory_tester.get_memory_chip());
    memory_tester.install(1, 0, [BabyBear::zero()]);
    memory_tester.install(1, 1, [BabyBear::one()]);

    execution_tester.execute(
        &mut field_arithmetic_chip,
        Instruction::from_usize(FDIV, 0, 0, 1, 1, 1),
    );

    let trace = field_arithmetic_chip.generate_trace();
    let row = trace.row_slice(0);
    let cols: &FieldArithmeticCols<_> = (*row).borrow();
    let mut cols = cols.clone();
    cols.io.y = BabyBear::zero();
    let trace = RowMajorMatrix::new(cols.flatten(), FieldArithmeticCols::<BabyBear>::get_width());

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![field_arithmetic_chip.air().deref()], vec![trace],),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[should_panic]
#[test]
fn au_air_test_panic() {
    let execution_bus = ExecutionBus(0);
    let memory_bus = 1;
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(execution_bus, memory_tester.get_memory_chip());
    memory_tester.install(1, 0, [BabyBear::zero()]);
    // should panic
    execution_tester.execute(
        &mut field_arithmetic_chip,
        Instruction::from_usize(FDIV, 0, 0, 0, 1, 1),
    );
}
