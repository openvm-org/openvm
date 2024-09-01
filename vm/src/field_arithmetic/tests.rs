use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::{baby_bear_poseidon2::run_simple_test_no_pis, setup_tracing},
    utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::Rng;

use super::{FieldArithmetic, FieldArithmeticChip};
use crate::{
    arch::{
        chips::MachineChip,
        instructions::{Opcode::*, FIELD_ARITHMETIC_INSTRUCTIONS},
        testing::MachineChipTestBuilder,
    },
    cpu::trace::Instruction,
    field_arithmetic::columns::{FieldArithmeticCols, FieldArithmeticIoCols},
};

#[test]
fn field_arithmetic_air_test() {
    setup_tracing();
    let num_ops = 3; // non-power-of-2 to also test padding
    let elem_range = || 1..=100;
    let z_address_space_range = || 1usize..=2;
    let xy_address_space_range = || 0usize..=2;
    let address_range = || 0usize..1 << 29;

    let mut tester = MachineChipTestBuilder::default();
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(tester.execution_bus(), tester.memory_chip());

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = FIELD_ARITHMETIC_INSTRUCTIONS[rng.gen_range(0..4)];

        let operand1 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));
        let operand2 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));

        if opcode == FDIV && operand2.is_zero() {
            continue;
        }

        let result_as = rng.gen_range(z_address_space_range());
        let as1 = rng.gen_range(xy_address_space_range());
        let as2 = rng.gen_range(xy_address_space_range());
        let address1 = if as1 == 0 {
            operand1.as_canonical_u32() as usize
        } else {
            rng.gen_range(address_range())
        };
        let address2 = if as2 == 0 {
            operand2.as_canonical_u32() as usize
        } else {
            rng.gen_range(address_range())
        };
        assert_ne!(address1, address2);
        let result_address = rng.gen_range(address_range());

        let result = FieldArithmetic::solve(opcode, (operand1, operand2)).unwrap();
        tracing::debug!(
            "{opcode} d = {}, e = {}, f = {}, result_addr = {}, addr1 = {}, addr2 = {}, z = {}, x = {}, y = {}",
            result_as, as1, as2, result_address, address1, address2, result, operand1, operand2,
        );

        if as1 != 0 {
            tester.write_cell(as1, address1, operand1);
        }
        if as2 != 0 {
            tester.write_cell(as2, address2, operand2);
        }
        tester.execute(
            &mut field_arithmetic_chip,
            Instruction::from_usize(
                opcode,
                [result_address, address1, address2, result_as, as1, as2],
            ),
        );
        assert_eq!(result, tester.read_cell(result_as, result_address));
    }

    // positive test
    let mut tester = tester.build().load(field_arithmetic_chip).finalize();

    tester.simple_test().expect("Verification failed");

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    // negative test pranking each IO value
    for height in 0..num_ops {
        // TODO: better way to modify existing traces in tester
        let arith_trace = &mut tester.traces[1];
        let old_trace = arith_trace.clone();
        for width in 0..FieldArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            arith_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        assert_eq!(
            tester.simple_test(),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );

        tester.traces[1] = old_trace;
    }
}

#[test]
fn field_arithmetic_air_zero_div_zero() {
    let mut tester = MachineChipTestBuilder::default();
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(tester.execution_bus(), tester.memory_chip());
    tester.write_cell(1, 0, BabyBear::zero());
    tester.write_cell(1, 1, BabyBear::one());

    tester.execute(
        &mut field_arithmetic_chip,
        Instruction::from_usize(FDIV, [0, 0, 1, 1, 1, 1]),
    );

    let air = field_arithmetic_chip.air;
    let trace = field_arithmetic_chip.generate_trace();
    let row = trace.row_slice(0).to_vec();
    let mut cols = FieldArithmeticCols::from_iter(&mut row.into_iter(), &air);
    cols.io.y.value = BabyBear::zero();
    let trace = RowMajorMatrix::new(
        cols.flatten(),
        FieldArithmeticCols::<BabyBear>::get_width(&air),
    );

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&air], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[should_panic]
#[test]
fn field_arithmetic_air_test_panic() {
    let mut tester = MachineChipTestBuilder::default();
    let mut field_arithmetic_chip =
        FieldArithmeticChip::new(tester.execution_bus(), tester.memory_chip());
    tester.write_cell(1, 0, BabyBear::zero());
    // should panic
    tester.execute(
        &mut field_arithmetic_chip,
        Instruction::from_usize(FDIV, [0, 0, 0, 1, 1, 1]),
    );
}
