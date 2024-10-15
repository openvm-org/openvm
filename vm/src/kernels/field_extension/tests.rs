use std::{
    array,
    ops::{Add, Div, Mul, Sub},
};

use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
use rand::Rng;
use strum::EnumCount;

use super::columns::FieldExtensionArithmeticIoCols;
use crate::{
    arch::{
        instructions::{FieldExtensionOpcode, UsizeOpcode},
        testing::{
            memory::{gen_address_space, gen_pointer},
            VmChipTestBuilder,
        },
    },
    kernels::field_extension::chip::{FieldExtensionArithmetic, FieldExtensionArithmeticChip},
    system::program::Instruction,
};

#[test]
fn field_extension_air_test() {
    type F = BabyBear;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = FieldExtensionArithmeticChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        0,
    );

    let mut rng = create_seeded_rng();
    let num_ops: usize = 7; // test padding with dummy row

    for _ in 0..num_ops {
        let opcode =
            FieldExtensionOpcode::from_usize(rng.gen_range(0..FieldExtensionOpcode::COUNT));

        let as_d = gen_address_space(&mut rng);
        let as_e = gen_address_space(&mut rng);
        let address1 = gen_pointer(&mut rng, 4);
        let address2 = gen_pointer(&mut rng, 4);
        let result_address = gen_pointer(&mut rng, 4);

        let operand1 = array::from_fn(|_| rng.gen::<F>());
        let operand2 = array::from_fn(|_| rng.gen::<F>());

        assert!(address1.abs_diff(address2) >= 4);

        tester.write(as_d, address1, operand1);
        tester.write(as_e, address2, operand2);

        let result = FieldExtensionArithmetic::solve(opcode, operand1, operand2).unwrap();

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                opcode as usize,
                [result_address, address1, address2, as_d, as_e],
            ),
        );
        assert_eq!(result, tester.read(as_d, result_address));
    }

    // positive test
    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    // negative test pranking each IO value
    for height in [0, num_ops - 1] {
        // TODO: better way to modify existing traces in tester
        let extension_trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
        let original_trace = extension_trace.clone();
        for width in 0..FieldExtensionArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            extension_trace.row_mut(height)[width] = prank_value;
        }

        assert_eq!(
            tester.simple_test().err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        tester.air_proof_inputs[2].raw.common_main = Some(original_trace);
    }
}

#[test]
fn field_extension_consistency_test() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let len_tests = 100;
    let mut rng = create_seeded_rng();

    let operands: Vec<([F; 4], [F; 4])> = (0..len_tests)
        .map(|_| {
            (
                array::from_fn(|_| rng.gen::<F>()),
                array::from_fn(|_| rng.gen::<F>()),
            )
        })
        .collect();

    for (a, b) in operands {
        let a_ext = EF::from_base_slice(&a);
        let b_ext = EF::from_base_slice(&b);

        let plonky_add = a_ext.add(b_ext);
        let plonky_sub = a_ext.sub(b_ext);
        let plonky_mul = a_ext.mul(b_ext);
        let plonky_div = a_ext.div(b_ext);

        let my_add = FieldExtensionArithmetic::add(a, b);
        let my_sub = FieldExtensionArithmetic::subtract(a, b);
        let my_mul = FieldExtensionArithmetic::multiply(a, b);
        let my_div = FieldExtensionArithmetic::divide(a, b);

        assert_eq!(my_add, plonky_add.as_base_slice());
        assert_eq!(my_sub, plonky_sub.as_base_slice());
        assert_eq!(my_mul, plonky_mul.as_base_slice());
        assert_eq!(my_div, plonky_div.as_base_slice());
    }
}
