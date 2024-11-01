use ax_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_stark_sdk::utils::create_seeded_rng;
use axvm_instructions::{instruction::Instruction, FriFoldOpcode::FRI_FOLD};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use rand::Rng;

use crate::{
    arch::testing::{memory::gen_pointer, VmChipTestBuilder},
    kernels::fri::{FriFoldChip, FriFoldCols},
};

fn compute_fri_fold<F: Field>(alpha: F, mut alpha_pow: F, a: &[F], b: &[F]) -> (F, F) {
    let mut result = F::zero();
    for (&a, &b) in a.iter().zip_eq(b) {
        result += (b - a) * alpha_pow;
        alpha_pow *= alpha;
    }
    (alpha_pow, result)
}

#[test]
fn fri_fold_air_test() {
    let num_ops = 3; // non-power-of-2 to also test padding
    let elem_range = || 1..=100;
    let address_space_range = || 1usize..=2;
    let length_range = || 1..=49;

    let mut tester = VmChipTestBuilder::default();
    let mut chip = FriFoldChip::new(
        tester.memory_controller(),
        tester.execution_bus(),
        tester.program_bus(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let alpha = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));
        let length = rng.gen_range(length_range());
        let alpha_pow_initial = BabyBear::from_canonical_u32(rng.gen_range(elem_range()));
        let a: Vec<BabyBear> = (0..length)
            .map(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())))
            .collect();
        let b: Vec<BabyBear> = (0..length)
            .map(|_| BabyBear::from_canonical_u32(rng.gen_range(elem_range())))
            .collect();

        let (alpha_pow_final, result) = compute_fri_fold(alpha, alpha_pow_initial, &a, &b);

        let alpha_pointer = gen_pointer(&mut rng, 1);
        let length_pointer = gen_pointer(&mut rng, 1);
        let alpha_pow_pointer = gen_pointer(&mut rng, 1);
        let result_pointer = gen_pointer(&mut rng, 1);
        let a_pointer = gen_pointer(&mut rng, length);
        let b_pointer = gen_pointer(&mut rng, length);

        let address_space = rng.gen_range(address_space_range());

        /*tracing::debug!(
            "{opcode:?} d = {}, e = {}, f = {}, result_addr = {}, addr1 = {}, addr2 = {}, z = {}, x = {}, y = {}",
            result_as, as1, as2, result_pointer, address1, address2, result, operand1, operand2,
        );*/

        tester.write_cell(address_space, alpha_pointer, alpha);
        tester.write_cell(
            address_space,
            length_pointer,
            BabyBear::from_canonical_usize(length),
        );
        tester.write_cell(address_space, alpha_pow_pointer, alpha_pow_initial);
        for i in 0..length {
            tester.write_cell(address_space, a_pointer + i, a[i]);
            tester.write_cell(address_space, b_pointer + i, b[i]);
        }

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                FRI_FOLD as usize,
                [
                    a_pointer,
                    b_pointer,
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
            tester.read_cell(address_space, alpha_pow_pointer)
        );
        assert_eq!(result, tester.read_cell(address_space, result_pointer));
    }

    let mut tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");

    disable_debug_builder();
    // negative test pranking each value
    for height in 0..num_ops {
        // TODO: better way to modify existing traces in tester
        let trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
        let old_trace = trace.clone();
        for width in 0..FriFoldCols::<BabyBear>::width()
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
