use itertools::Itertools;
use openvm_circuit::arch::testing::{memory::gen_pointer, VmChipTestBuilder};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::FriOpcode::FRI_REDUCED_OPENING;
use openvm_stark_backend::{
    p3_field::{Field, FieldAlgebra},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::Rng;

use super::{
    super::field_extension::FieldExtension, elem_to_ext, FriReducedOpeningAir,
    FriReducedOpeningChip, FriReducedOpeningStep, EXT_DEG,
};
use crate::fri::OVERALL_WIDTH;

const MAX_INS_CAPACITY: usize = 1024;
type F = BabyBear;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> FriReducedOpeningChip<F> {
    let mut chip = FriReducedOpeningChip::<F>::new(
        FriReducedOpeningAir::new(tester.execution_bridge(), tester.memory_bridge()),
        FriReducedOpeningStep::new(),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    chip
}

fn compute_fri_mat_opening<F: Field>(
    alpha: [F; EXT_DEG],
    a: &[F],
    b: &[[F; EXT_DEG]],
) -> [F; EXT_DEG] {
    let mut alpha_pow: [F; EXT_DEG] = elem_to_ext(F::ONE);
    let mut result = [F::ZERO; EXT_DEG];
    for (&a, &b) in a.iter().zip_eq(b) {
        result = FieldExtension::add(
            result,
            FieldExtension::multiply(FieldExtension::subtract(b, elem_to_ext(a)), alpha_pow),
        );
        alpha_pow = FieldExtension::multiply(alpha, alpha_pow);
    }
    result
}

#[test]
fn fri_mat_opening_air_test() {
    let num_ops = 14; // non-power-of-2 to also test padding
    let elem_range = || 1..=100;
    let length_range = || 1..=49;

    let mut tester = VmChipTestBuilder::default_native();

    let mut chip = create_test_chip(&tester);

    let mut rng = create_seeded_rng();

    macro_rules! gen_ext {
        () => {
            std::array::from_fn::<_, EXT_DEG, _>(|_| {
                BabyBear::from_canonical_u32(rng.gen_range(elem_range()))
            })
        };
    }

    tester.streams.hint_space = vec![vec![]];

    for _ in 0..num_ops {
        let alpha = gen_ext!();
        let length = rng.gen_range(length_range());
        let a = (0..length)
            .map(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())))
            .collect_vec();
        let b = (0..length).map(|_| gen_ext!()).collect_vec();

        let result = compute_fri_mat_opening(alpha, &a, &b);

        let alpha_pointer = gen_pointer(&mut rng, 4);
        let length_pointer = gen_pointer(&mut rng, 1);
        let a_pointer_pointer = gen_pointer(&mut rng, 1);
        let b_pointer_pointer = gen_pointer(&mut rng, 1);
        let result_pointer = gen_pointer(&mut rng, 4);
        let a_pointer = gen_pointer(&mut rng, 1);
        let b_pointer = gen_pointer(&mut rng, 4);
        let is_init_ptr = gen_pointer(&mut rng, 1);

        let address_space = 4usize;

        // tracing::debug!(
        //     "{opcode:?} d = {}, e = {}, f = {}, result_addr = {}, addr1 = {}, addr2 = {}, z = {},
        // x = {}, y = {}",     result_as, as1, as2, result_pointer, address1, address2,
        // result, operand1, operand2, );

        tester.write(address_space, alpha_pointer, alpha);
        tester.write(
            address_space,
            length_pointer,
            [BabyBear::from_canonical_usize(length)],
        );
        tester.write(
            address_space,
            a_pointer_pointer,
            [BabyBear::from_canonical_usize(a_pointer)],
        );
        tester.write(
            address_space,
            b_pointer_pointer,
            [BabyBear::from_canonical_usize(b_pointer)],
        );
        let is_init = rng.gen_range(0..2);
        tester.write(
            address_space,
            is_init_ptr,
            [BabyBear::from_canonical_u32(is_init)],
        );

        if is_init == 0 {
            tester.streams.hint_space[0].extend_from_slice(&a);
        } else {
            for (i, ai) in a.iter().enumerate() {
                tester.write(address_space, a_pointer + i, [*ai]);
            }
        }
        for (i, bi) in b.iter().enumerate() {
            tester.write(address_space, b_pointer + (4 * i), *bi);
        }

        tester.execute(
            &mut chip,
            &Instruction::from_usize(
                FRI_REDUCED_OPENING.global_opcode(),
                [
                    a_pointer_pointer,
                    b_pointer_pointer,
                    length_pointer,
                    alpha_pointer,
                    result_pointer,
                    0, // hint id
                    is_init_ptr,
                ],
            ),
        );
        assert_eq!(result, tester.read(address_space, result_pointer));
        // Check that `a` was populated.
        for (i, ai) in a.iter().enumerate() {
            let [found] = tester.read(address_space, a_pointer + i);
            assert_eq!(*ai, found);
        }
    }

    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    // negative test pranking each value
    for height in 0..num_ops {
        // TODO: better way to modify existing traces in tester
        let trace = tester.air_proof_inputs[2]
            .1
            .raw
            .common_main
            .as_mut()
            .unwrap();
        let old_trace = trace.clone();
        for width in 0..OVERALL_WIDTH
        /* num operands */
        {
            let prank_value = BabyBear::from_canonical_u32(rng.gen_range(1..=100));
            trace.row_mut(height)[width] = prank_value;
        }

        // Run a test after pranking each row
        assert_eq!(
            tester.simple_test().err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );

        tester.air_proof_inputs[2].1.raw.common_main = Some(old_trace);
    }
}
