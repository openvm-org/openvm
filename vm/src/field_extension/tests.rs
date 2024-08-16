use std::ops::{Add, Div, Mul, Sub};

use p3_baby_bear::BabyBear;
use p3_field::{AbstractExtensionField, AbstractField, extension::BinomialExtensionField};
use rand::Rng;

use afs_stark_backend::verifier::VerificationError;
use afs_test_utils::utils::create_seeded_rng;

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::MachineChip,
        testing::{ExecutionTester, MachineChipTester, MemoryTester},
    },
    cpu::{FIELD_EXTENSION_INSTRUCTIONS, OpCode, trace::Instruction},
};

use super::{
    columns::FieldExtensionArithmeticIoCols, FieldExtensionArithmeticAir,
    FieldExtensionArithmeticChip,
};

#[test]
fn field_extension_air_test() {
    let num_ops = 13;
    let elem_range = || 1..=100;
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 30;

    let execution_bus = ExecutionBus(0);
    let memory_bus = 1;
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut field_extension_chip =
        FieldExtensionArithmeticChip::new(execution_bus, memory_tester.get_memory_chip());

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = FIELD_EXTENSION_INSTRUCTIONS[rng.gen_range(0..4)];

        let operand1 =
            std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())));
        let operand2 =
            std::array::from_fn(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())));

        let as_d = rng.gen_range(address_space_range());
        let as_e = rng.gen_range(address_space_range());
        let address1 = rng.gen_range(address_range());
        let address2 = rng.gen_range(address_range());
        let result_address = rng.gen_range(address_range());
        assert!(address1.abs_diff(address2) >= 4);

        let result = FieldExtensionArithmeticAir::solve(opcode, operand1, operand2).unwrap();

        memory_tester.install(as_d, address1, operand1);
        memory_tester.install(as_e, address2, operand2);
        execution_tester.execute(
            &mut field_extension_chip,
            Instruction::from_usize(opcode, result_address, address1, address2, as_d, as_e),
        );
        memory_tester.expect(as_d, result_address, result);
        memory_tester.check();
    }

    MachineChipTester::default()
        .add(&mut execution_tester)
        .add(&mut memory_tester)
        .add(&mut field_extension_chip)
        .simple_test()
        .expect("Verification failed");

    // negative test pranking each IO value
    for height in 0..num_ops {
        let mut extension_trace = field_extension_chip.generate_trace();
        for width in 0..FieldExtensionArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            extension_trace.row_mut(height)[width] = prank_value;
        }

        let test_result = MachineChipTester::default()
            .add(&mut execution_tester)
            .add(&mut memory_tester)
            .add_with_custom_trace(&mut field_extension_chip, extension_trace)
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
fn field_extension_consistency_test() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let len_tests = 100;
    let mut rng = create_seeded_rng();

    let operands: Vec<([F; 4], [F; 4])> = (0..len_tests)
        .map(|_| {
            (
                [
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                ],
                [
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                    F::from_canonical_u32(rng.gen_range(1..=100)),
                ],
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

        let my_add = FieldExtensionArithmeticAir::solve(OpCode::FE4ADD, a, b);
        let my_sub = FieldExtensionArithmeticAir::solve(OpCode::FE4SUB, a, b);
        let my_mul = FieldExtensionArithmeticAir::solve(OpCode::BBE4MUL, a, b);

        let b_inv = FieldExtensionArithmeticAir::solve(OpCode::BBE4INV, b, [F::zero(); 4]).unwrap();
        let my_div = FieldExtensionArithmeticAir::solve(OpCode::BBE4MUL, a, b_inv);

        assert_eq!(my_add.unwrap(), plonky_add.as_base_slice());
        assert_eq!(my_sub.unwrap(), plonky_sub.as_base_slice());
        assert_eq!(my_mul.unwrap(), plonky_mul.as_base_slice());
        assert_eq!(my_div.unwrap(), plonky_div.as_base_slice());
    }
}
