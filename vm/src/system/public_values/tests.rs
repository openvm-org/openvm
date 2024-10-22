use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError, Chip};
use ax_sdk::{config::baby_bear_poseidon2::BabyBearPoseidon2Config, utils::to_field_vec};
use axvm_instructions::PublishOpcode;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};

use crate::{
    arch::testing::VmChipTestBuilder,
    kernels::adapters::native_adapter::NativeAdapterChip,
    system::{
        program::Instruction,
        public_values::{core::PublicValuesCoreChip, PublicValuesChip},
    },
};

type F = BabyBear;
type AdapterChip<F> = NativeAdapterChip<F, 2, 0>;

fn setup_test_chip<F: PrimeField32>(
    tester: &mut VmChipTestBuilder<F>,
    num_pvs: usize,
) -> PublicValuesChip<F> {
    PublicValuesChip::new(
        AdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        PublicValuesCoreChip::new(num_pvs, 0),
        tester.memory_controller(),
    )
}

#[test]
fn public_values_happy_path() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = setup_test_chip(&mut tester, 3);
    {
        tester.execute(
            &mut chip,
            Instruction::from_usize(PublishOpcode::PUBLISH as usize, [0, 12, 2, 0, 0, 0]),
        );
    }

    let mut air_input = Chip::<BabyBearPoseidon2Config>::generate_air_proof_input(chip);
    assert_eq!(air_input.raw.public_values, to_field_vec(vec![0, 0, 12]));
    // If not specified, the public value could be any value.
    air_input.raw.public_values[1] = F::from_canonical_u32(5456789);

    let tester = tester.build().load_air_proof_input(air_input).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn public_values_neg_pv_not_match() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = setup_test_chip(&mut tester, 3);
    {
        tester.execute(
            &mut chip,
            Instruction::from_usize(PublishOpcode::PUBLISH as usize, [0, 12, 2, 0, 0, 0]),
        );
    }

    let mut air_input = Chip::<BabyBearPoseidon2Config>::generate_air_proof_input(chip);
    assert_eq!(air_input.raw.public_values, to_field_vec(vec![0, 0, 12]));
    // Set public value to a different value.
    air_input.raw.public_values[2] = F::from_canonical_u32(5456789);

    disable_debug_builder();
    let tester = tester.build().load_air_proof_input(air_input).finalize();
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

#[test]
fn public_values_neg_index_out_of_bound() {
    let mut tester = VmChipTestBuilder::default();
    let mut chip = setup_test_chip(&mut tester, 3);
    {
        // [Negative] The public value index is 8 which is out of bounds.
        tester.execute(
            &mut chip,
            Instruction::from_usize(PublishOpcode::PUBLISH as usize, [0, 12, 8, 0, 0, 0]),
        );
    }

    let air_input = Chip::<BabyBearPoseidon2Config>::generate_air_proof_input(chip);
    assert_eq!(air_input.raw.public_values, to_field_vec(vec![0, 0, 0]));

    disable_debug_builder();
    let tester = tester.build().load_air_proof_input(air_input).finalize();
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

#[test]
fn public_values_neg_double_publish() {
    for i in 0..1 {
        let mut tester = VmChipTestBuilder::default();
        let mut chip = setup_test_chip(&mut tester, 3);
        {
            // [Negative] The 2nd public values are published twice with different values.
            tester.execute(
                &mut chip,
                Instruction::from_usize(PublishOpcode::PUBLISH as usize, [0, 12, 2, 0, 0, 0]),
            );
            tester.execute(
                &mut chip,
                Instruction::from_usize(PublishOpcode::PUBLISH as usize, [0, 13, 2, 0, 0, 0]),
            );
        }

        let mut air_input = Chip::<BabyBearPoseidon2Config>::generate_air_proof_input(chip);
        assert_eq!(air_input.raw.public_values, to_field_vec(vec![0, 0, 12]));
        // No matter what the public value is 12 or 13, the proof should fail.
        if i == 1 {
            air_input.raw.public_values[2] = F::from_canonical_u32(13);
        }

        disable_debug_builder();
        let tester = tester.build().load_air_proof_input(air_input).finalize();
        assert_eq!(
            tester.simple_test().err(),
            Some(VerificationError::OodEvaluationMismatch)
        );
    }
}
