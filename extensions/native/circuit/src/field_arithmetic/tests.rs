use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::testing::{memory::gen_pointer, VmChipTestBuilder},
    system::native_adapter::{NativeAdapterChip, NativeAdapterCols},
};
use openvm_instructions::{instruction::Instruction, UsizeOpcode, VmOpcode};
use openvm_native_compiler::FieldArithmeticOpcode;
use openvm_stark_backend::{
    p3_field::{Field, FieldAlgebra, PrimeField32},
    prover::USE_DEBUG_BUILDER,
    utils::disable_debug_builder,
    verifier::VerificationError,
    Chip,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::Rng;
use strum::EnumCount;

use super::{
    core::FieldArithmeticCoreChip, FieldArithmetic, FieldArithmeticChip, FieldArithmeticCoreCols,
};

#[test]
fn new_field_arithmetic_air_test() {
    let num_ops = 3; // non-power-of-2 to also test padding
    let elem_range = || 1..=100;
    let z_address_space_range = || 1usize..=2;
    let xy_address_space_range = || 0usize..=2;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = FieldArithmeticChip::new(
        NativeAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
        ),
        FieldArithmeticCoreChip::new(0),
        tester.offline_memory_mutex_arc(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode =
            FieldArithmeticOpcode::from_usize(rng.gen_range(0..FieldArithmeticOpcode::COUNT));

        let operand1 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));
        let operand2 = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));

        if opcode == FieldArithmeticOpcode::DIV && operand2.is_zero() {
            continue;
        }

        let result_as = rng.gen_range(z_address_space_range());
        let as1 = rng.gen_range(xy_address_space_range());
        let as2 = rng.gen_range(xy_address_space_range());
        let address1 = if as1 == 0 {
            operand1.as_canonical_u32() as usize
        } else {
            gen_pointer(&mut rng, 1)
        };
        let address2 = if as2 == 0 {
            operand2.as_canonical_u32() as usize
        } else {
            gen_pointer(&mut rng, 1)
        };
        assert_ne!(address1, address2);
        let result_address = gen_pointer(&mut rng, 1);

        let result = FieldArithmetic::run_field_arithmetic(opcode, operand1, operand2).unwrap();
        tracing::debug!(
            "{opcode:?} d = {}, e = {}, f = {}, result_addr = {}, addr1 = {}, addr2 = {}, z = {}, x = {}, y = {}",
            result_as, as1, as2, result_address, address1, address2, result, operand1, operand2,
        );

        if as1 != 0 {
            tester.write_cell(as1, address1, operand1);
        }
        if as2 != 0 {
            tester.write_cell(as2, address2, operand2);
        }
        tester.execute(
            &mut chip,
            Instruction::from_usize(
                VmOpcode::from_usize(opcode as usize),
                [result_address, address1, address2, result_as, as1, as2],
            ),
        );
        assert_eq!(result, tester.read_cell(result_as, result_address));
    }

    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    // negative test pranking each IO value
    for height in 0..num_ops {
        // TODO: better way to modify existing traces in tester
        let arith_trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
        let old_trace = arith_trace.clone();
        for width in 0..FieldArithmeticCoreCols::<BabyBear>::width() {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            arith_trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        assert_eq!(
            tester.simple_test().err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );

        tester.air_proof_inputs[2].raw.common_main = Some(old_trace);
    }
}

#[test]
fn new_field_arithmetic_air_zero_div_zero() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = FieldArithmeticChip::new(
        NativeAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
        ),
        FieldArithmeticCoreChip::new(0),
        tester.offline_memory_mutex_arc(),
    );
    tester.write_cell(1, 6, BabyBear::from_canonical_u32(111));
    tester.write_cell(1, 7, BabyBear::from_canonical_u32(222));

    tester.execute(
        &mut chip,
        Instruction::from_usize(
            VmOpcode::from_usize(FieldArithmeticOpcode::DIV as usize),
            [5, 6, 7, 1, 1, 1],
        ),
    );
    tester.build();

    let mut chip_input = chip.generate_air_proof_input();
    // set the value of [c]_f to zero, necessary to bypass trace gen checks
    let row = chip_input.raw.common_main.as_mut().unwrap().row_mut(0);
    let cols: &mut FieldArithmeticCoreCols<BabyBear> = row
        .split_at_mut(NativeAdapterCols::<BabyBear, 2, 1>::width())
        .1
        .borrow_mut();
    cols.b = BabyBear::ZERO;

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });

    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(vec![chip_input]).err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[should_panic]
#[test]
fn new_field_arithmetic_air_test_panic() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = FieldArithmeticChip::new(
        NativeAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
        ),
        FieldArithmeticCoreChip::new(0),
        tester.offline_memory_mutex_arc(),
    );
    tester.write_cell(1, 0, BabyBear::ZERO);
    // should panic
    tester.execute(
        &mut chip,
        Instruction::from_usize(
            VmOpcode::from_usize(FieldArithmeticOpcode::DIV as usize),
            [0, 0, 0, 1, 1, 1],
        ),
    );
}
