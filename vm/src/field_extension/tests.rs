use std::{
    array,
    ops::{Add, Div, Mul, Sub},
};

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
use rand::Rng;

use super::columns::FieldExtensionArithmeticIoCols;
use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::MachineChip,
        instructions::{Opcode, FIELD_EXTENSION_INSTRUCTIONS},
        testing::{ExecutionTester, MachineChipTester, MemoryTester},
    },
    field_extension::chip::{
        FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticRecord,
    },
    memory::{manager::MemoryAccess, offline_checker::bus::MemoryBus},
};

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_records(n: usize) -> Vec<FieldExtensionArithmeticRecord<BabyBear>> {
    let mut rng = create_seeded_rng();

    let mut records = vec![];

    for _ in 0..n {
        let opcode = FIELD_EXTENSION_INSTRUCTIONS[rng.gen_range(0..4)];

        // dummy values for clock cycle and addr_space and pointers
        let timestamp: usize = 1;

        let x = [
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
        ];
        let y = [
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
        ];

        let z = FieldExtensionArithmetic::solve(opcode, x, y).unwrap();

        records.push(FieldExtensionArithmeticRecord {
            pc: 0,
            timestamp,
            opcode,
            is_valid: false,
            op_a: BabyBear::zero(),
            op_b: BabyBear::zero(),
            op_c: BabyBear::zero(),
            d: BabyBear::one(),
            e: BabyBear::one(),
            x,
            y,
            z,
            x_reads: array::from_fn(|_| {
                MemoryAccess::disabled_read(
                    BabyBear::from_canonical_usize(timestamp),
                    BabyBear::one(),
                )
            }),
            y_reads: array::from_fn(|_| {
                MemoryAccess::disabled_read(
                    BabyBear::from_canonical_usize(timestamp),
                    BabyBear::one(),
                )
            }),
            z_writes: array::from_fn(|_| {
                MemoryAccess::disabled_read(
                    BabyBear::from_canonical_usize(timestamp),
                    BabyBear::one(),
                )
            }),
        });
    }
    records
}

#[test]
fn field_extension_air_test() {
    let num_ops = 13;
    let elem_range = || 1..=100;
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let execution_bus = ExecutionBus(0);
    let memory_bus = MemoryBus(1);
    let mut execution_tester = ExecutionTester::new(execution_bus, create_seeded_rng());
    let mut memory_tester = MemoryTester::new(memory_bus);
    let mut field_extension_chip =
        FieldExtensionArithmeticChip::new(execution_bus, memory_tester.get_memory_manager());

    let mut rng = create_seeded_rng();
    let num_ops: usize = 1 << 5;

    let mut chip = FieldExtensionArithmeticChip::<BabyBear>::new(
        execution_bus,
        memory_tester.get_memory_manager(),
    );
    chip.records = generate_records(num_ops);

    // for _ in 0..num_ops {
    //     let opcode = FIELD_EXTENSION_INSTRUCTIONS[rng.gen_range(0..4)];

    //     let as_d = rng.gen_range(address_space_range());
    //     let as_e = rng.gen_range(address_space_range());
    //     let address1 = rng.gen_range(address_range());
    //     let address2 = rng.gen_range(address_range());
    //     let result_address = rng.gen_range(address_range());
    //     assert!(address1.abs_diff(address2) >= 4);
    //     println!(
    //         "d = {}, e = {}, result_addr = {}, addr1 = {}, addr2 = {}",
    //         as_d, as_e, result_address, address1, address2
    //     );

    //     let result = FieldExtensionArithmetic::solve(opcode, operand1, operand2).unwrap();

    //     memory_tester.install(as_d, address1, operand1);
    //     memory_tester.install(as_e, address2, operand2);
    //     execution_tester.execute(
    //         &mut memory_tester,
    //         &mut field_extension_chip,
    //         Instruction::from_usize(opcode, [result_address, address1, address2, as_d, as_e]),
    //     );
    //     memory_tester.expect(as_d, result_address, result);
    //     memory_tester.check();
    // }

    // positive test
    MachineChipTester::default()
        .add(&mut execution_tester)
        .add(&mut memory_tester)
        .add(&mut field_extension_chip)
        .simple_test()
        .expect("Verification failed");

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

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

        let my_add = FieldExtensionArithmetic::solve(Opcode::FE4ADD, a, b);
        let my_sub = FieldExtensionArithmetic::solve(Opcode::FE4SUB, a, b);
        let my_mul = FieldExtensionArithmetic::solve(Opcode::BBE4MUL, a, b);

        let b_inv = FieldExtensionArithmetic::solve(Opcode::BBE4INV, b, [F::zero(); 4]).unwrap();
        let my_div = FieldExtensionArithmetic::solve(Opcode::BBE4MUL, a, b_inv);

        assert_eq!(my_add.unwrap(), plonky_add.as_base_slice());
        assert_eq!(my_sub.unwrap(), plonky_sub.as_base_slice());
        assert_eq!(my_mul.unwrap(), plonky_mul.as_base_slice());
        assert_eq!(my_div.unwrap(), plonky_div.as_base_slice());
    }
}
