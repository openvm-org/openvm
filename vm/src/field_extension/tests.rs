use std::{
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, AbstractField};
use p3_matrix::Matrix;
use rand::Rng;

use super::columns::FieldExtensionArithmeticIoCols;
use crate::{
    cpu::{OpCode, FIELD_EXTENSION_INSTRUCTIONS, RANGE_CHECKER_BUS, WORD_SIZE},
    field_extension::chip::{
        FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticRecord,
    },
    memory::manager::MemoryManager,
    vm::config::MemoryConfig,
};

/// Function for testing that generates a random program consisting only of field arithmetic operations.
fn generate_field_extension_operations(
    len_ops: usize,
) -> Vec<FieldExtensionArithmeticRecord<1, BabyBear>> {
    let mut rng = create_seeded_rng();

    let mut requests = vec![];

    for _ in 0..len_ops {
        let op = FIELD_EXTENSION_INSTRUCTIONS[rng.gen_range(0..4)];

        // dummy values for clock cycle and addr_space and pointers
        let timestamp: usize = 0;
        let op_a = BabyBear::zero();
        let op_b = BabyBear::zero();
        let op_c = BabyBear::zero();
        let d = BabyBear::zero();
        let e = BabyBear::zero();

        let operand1 = [
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
        ];
        let operand2 = [
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
            BabyBear::from_canonical_u32(rng.gen_range(1..=100)),
        ];

        let result = FieldExtensionArithmetic::solve(op, operand1, operand2).unwrap();

        requests.push(FieldExtensionArithmeticRecord {
            clk: timestamp,
            opcode: op,
            op_a,
            op_b,
            op_c,
            d,
            e,
            x: operand1,
            y: operand2,
            z: result,
        });
    }
    requests
}

// isolated air test
#[test]
#[ignore]
// TODO: rewrite this test
fn field_extension_air_test() {
    let mut rng = create_seeded_rng();
    let len_ops: usize = 1 << 5;

    let mem_config = MemoryConfig::new(16, 16, 16, 16);
    let range_checker = Arc::new(RangeCheckerGateChip::new(
        RANGE_CHECKER_BUS,
        (1 << mem_config.decomp) as u32,
    ));
    let memory_manager = Rc::new(RefCell::new(MemoryManager::with_volatile_memory(
        mem_config,
        range_checker.clone(),
    )));

    let mut chip = FieldExtensionArithmeticChip::<1, WORD_SIZE, BabyBear>::new(
        mem_config,
        memory_manager,
        range_checker,
    );
    let operations = generate_field_extension_operations(len_ops);
    chip.records = operations;

    let mut extension_trace = chip.generate_trace();

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    // positive test - should only error when interactions have nonzero cumulative sum
    assert_eq!(
        run_simple_test_no_pis(vec![&chip.air], vec![extension_trace.clone()]),
        Err(VerificationError::NonZeroCumulativeSum),
        "Expected constraint to pass"
    );

    // negative test pranking each IO value
    for height in 0..extension_trace.height() {
        for width in 0..FieldExtensionArithmeticIoCols::<BabyBear>::get_width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            extension_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        assert_eq!(
            run_simple_test_no_pis(vec![&chip.air], vec![extension_trace.clone()]),
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

        let my_add = FieldExtensionArithmetic::solve(OpCode::FE4ADD, a, b);
        let my_sub = FieldExtensionArithmetic::solve(OpCode::FE4SUB, a, b);
        let my_mul = FieldExtensionArithmetic::solve(OpCode::BBE4MUL, a, b);

        let b_inv = FieldExtensionArithmetic::solve(OpCode::BBE4INV, b, [F::zero(); 4]).unwrap();
        let my_div = FieldExtensionArithmetic::solve(OpCode::BBE4MUL, a, b_inv);

        assert_eq!(my_add.unwrap(), plonky_add.as_base_slice());
        assert_eq!(my_sub.unwrap(), plonky_sub.as_base_slice());
        assert_eq!(my_mul.unwrap(), plonky_mul.as_base_slice());
        assert_eq!(my_div.unwrap(), plonky_div.as_base_slice());
    }
}
