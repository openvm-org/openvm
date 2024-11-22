use ax_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_stark_sdk::utils::create_seeded_rng;
use axvm_instructions::{
    instruction::Instruction,
    FriOpcode::{self, FRI_REDUCED_OPENING},
    UsizeOpcode,
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use rand::Rng;

use crate::{
    arch::testing::{memory::gen_pointer, VmChipTestBuilder},
    kernels::{
        field_extension::FieldExtension,
        fri::{elem_to_ext, FriReducedOpeningChip, FriReducedOpeningCols, EXT_DEG},
    },
};

fn compute_fri_mat_opening<F: Field>(
    alpha: [F; EXT_DEG],
    mut alpha_pow: [F; EXT_DEG],
    a: &[F],
    b: &[[F; EXT_DEG]],
) -> ([F; EXT_DEG], [F; EXT_DEG]) {
    let mut result = [F::ZERO; EXT_DEG];
    for (&a, &b) in a.iter().zip_eq(b) {
        result = FieldExtension::add(
            result,
            FieldExtension::multiply(FieldExtension::subtract(b, elem_to_ext(a)), alpha_pow),
        );
        alpha_pow = FieldExtension::multiply(alpha, alpha_pow);
    }
    (alpha_pow, result)
}

#[test]
fn fri_mat_opening_air_test() {
    let num_ops = 3; // non-power-of-2 to also test padding
    let elem_range = || 1..=100;
    let address_space_range = || 1usize..=2;
    let length_range = || 1..=49;

    let offset = FriOpcode::default_offset();

    let mut tester = VmChipTestBuilder::default();
    let mut chip = FriReducedOpeningChip::new(
        tester.memory_controller(),
        tester.execution_bus(),
        tester.program_bus(),
        offset,
    );

    let mut rng = create_seeded_rng();

    macro_rules! gen_ext {
        () => {
            std::array::from_fn::<_, EXT_DEG, _>(|_| {
                BabyBear::from_canonical_u32(rng.gen_range(elem_range()))
            })
        };
    }

    for _ in 0..num_ops {
        let alpha = gen_ext!();
        let length = rng.gen_range(length_range());
        let alpha_pow_initial = gen_ext!();
        let a = (0..length)
            .map(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())))
            .collect_vec();
        let b = (0..length).map(|_| gen_ext!()).collect_vec();

        let (alpha_pow_final, result) = compute_fri_mat_opening(alpha, alpha_pow_initial, &a, &b);

        let alpha_pointer = gen_pointer(&mut rng, 4);
        let length_pointer = gen_pointer(&mut rng, 1);
        let a_pointer_pointer = gen_pointer(&mut rng, 1);
        let b_pointer_pointer = gen_pointer(&mut rng, 1);
        let alpha_pow_pointer = gen_pointer(&mut rng, 4);
        let result_pointer = gen_pointer(&mut rng, 4);
        let a_pointer = gen_pointer(&mut rng, 1);
        let b_pointer = gen_pointer(&mut rng, 4);

        let address_space = rng.gen_range(address_space_range());

        /*tracing::debug!(
            "{opcode:?} d = {}, e = {}, f = {}, result_addr = {}, addr1 = {}, addr2 = {}, z = {}, x = {}, y = {}",
            result_as, as1, as2, result_pointer, address1, address2, result, operand1, operand2,
        );*/

        tester.write(address_space, alpha_pointer, alpha);
        tester.write_cell(
            address_space,
            length_pointer,
            BabyBear::from_canonical_usize(length),
        );
        tester.write_cell(
            address_space,
            a_pointer_pointer,
            BabyBear::from_canonical_usize(a_pointer),
        );
        tester.write_cell(
            address_space,
            b_pointer_pointer,
            BabyBear::from_canonical_usize(b_pointer),
        );
        tester.write(address_space, alpha_pow_pointer, alpha_pow_initial);
        for i in 0..length {
            tester.write_cell(address_space, a_pointer + i, a[i]);
            tester.write(address_space, b_pointer + (4 * i), b[i]);
        }

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                (FRI_REDUCED_OPENING as usize) + offset,
                [
                    a_pointer_pointer,
                    b_pointer_pointer,
                    result_pointer,
                    address_space,
                    length_pointer,
                    alpha_pointer,
                    alpha_pow_pointer,
                ],
            ),
        );
        assert_eq!(
            alpha_pow_final,
            tester.read(address_space, alpha_pow_pointer)
        );
        assert_eq!(result, tester.read(address_space, result_pointer));
    }

    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    // negative test pranking each value
    for height in 0..num_ops {
        // TODO: better way to modify existing traces in tester
        let trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
        let old_trace = trace.clone();
        for width in 0..FriReducedOpeningCols::<BabyBear>::width()
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

        tester.air_proof_inputs[2].raw.common_main = Some(old_trace);
    }
}
