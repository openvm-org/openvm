use ark_ff::PrimeField as _;
use openvm_stark_backend::{
    p3_field::{AbstractField, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    any_rap_arc_vec,
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
    },
    dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    engine::StarkEngine,
    p3_baby_bear::{BabyBear, BabyBearInternalLayerParameters, Poseidon2BabyBear},
    utils::create_seeded_rng,
};
use p3_monty_31::InternalLayerBaseParameters;
use p3_poseidon2::{ExternalLayerConstants, Poseidon2};
use p3_symmetric::Permutation;
use rand::{Rng, RngCore};
use zkhash::{
    fields::babybear::FpBabyBear as HorizenBabyBear,
    poseidon2::{
        poseidon2::Poseidon2 as HorizenPoseidon2,
        poseidon2_instance_babybear::POSEIDON2_BABYBEAR_16_PARAMS,
    },
};

use super::{HL_BABYBEAR_EXT_CONST_16, HL_BABYBEAR_INT_CONST_16, HL_MDS_MAT_4, MDS_MAT_4};
use crate::poseidon2::Poseidon2Air;

#[test]
fn test_poseidon2_default() {
    // config
    let num_rows = 1 << 4;
    let num_ext_rounds = 8;

    // random constants, state generation
    let mut rng = create_seeded_rng();
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    // air and trace generation
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::default(); // max constraint degree = 7

    let mut poseidon2_trace = poseidon2_air.generate_trace(states.clone());
    let mut outputs = states.clone();
    let poseidon2: Poseidon2BabyBear<16> = Poseidon2::new(
        ExternalLayerConstants::new(
            HL_BABYBEAR_EXT_CONST_16[..num_ext_rounds / 2].to_vec(),
            HL_BABYBEAR_EXT_CONST_16[num_ext_rounds / 2..].to_vec(),
        ),
        HL_BABYBEAR_INT_CONST_16.to_vec(),
    );
    for output in outputs.iter_mut() {
        poseidon2.permute_mut(output);
    }

    // dummy interaction air and trace generation
    let page_requester = DummyInteractionAir::new(2 * 16, true, poseidon2_air.bus_index);
    let dummy_trace = RowMajorMatrix::new(
        states
            .into_iter()
            .zip(outputs.iter())
            .flat_map(|(state, output)| {
                [BabyBear::ONE]
                    .into_iter()
                    .chain(state.to_vec())
                    .chain(output.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect(),
        2 * 16 + 1,
    );

    let traces = vec![poseidon2_trace.clone(), dummy_trace.clone()];

    // engine generation
    let perm = random_perm();
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3); // max constraint degree = 7 requires log blowup = 3
    let engine = engine_from_perm(perm, fri_params);

    // positive test
    engine
        .run_simple_test_impl(
            any_rap_arc_vec![poseidon2_air.clone(), page_requester],
            traces,
            vec![vec![]; 2],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let width = rng.gen_range(0..poseidon2_air.get_width());
        let height = rng.gen_range(0..num_rows);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut(height)[width] += rand;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    any_rap_arc_vec![poseidon2_air.clone(), page_requester],
                    vec![poseidon2_trace.clone(), dummy_trace.clone()],
                    vec![vec![]; 2],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut(height)[width] -= rand;
    }
}

// Attention: if this test fails, it may be because plonky3 changed their constants.
// Check the reduction factor, which is either 1 or BabyBear::from_wrapped_u64(1u64 << 32).inverse(), // 943718400
#[test]
fn test_poseidon2() {
    // config
    let num_rows = 1 << 4;
    let num_ext_rounds = 8;
    let num_int_rounds = 13;

    // random constants, state generation
    let mut rng = create_seeded_rng();
    let external_constants = ExternalLayerConstants::new_from_rng(num_ext_rounds, &mut rng);
    let internal_constants: Vec<BabyBear> = (0..num_int_rounds)
        .map(|_| BabyBear::from_wrapped_u32(rng.next_u32()))
        .collect();
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    // air and trace generation
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::new(
        [
            &external_constants.get_initial_constants()[..],
            &external_constants.get_terminal_constants()[..],
        ]
        .concat(),
        internal_constants.clone(),
        MDS_MAT_4,
        BabyBearInternalLayerParameters::INTERNAL_DIAG_MONTY,
        BabyBear::ONE,
        3,
        0,
    );
    let mut poseidon2_trace = poseidon2_air.generate_trace(states.clone());
    let mut outputs = states.clone();
    let poseidon2: Poseidon2BabyBear<16> = Poseidon2::new(external_constants, internal_constants);
    for output in outputs.iter_mut() {
        poseidon2.permute_mut(output);
    }

    // dummy interaction air and trace generation
    let page_requester = DummyInteractionAir::new(2 * 16, true, poseidon2_air.bus_index);
    let dummy_trace = RowMajorMatrix::new(
        states
            .into_iter()
            .zip(outputs.iter())
            .flat_map(|(state, output)| {
                [BabyBear::ONE]
                    .into_iter()
                    .chain(state.to_vec())
                    .chain(output.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect(),
        2 * 16 + 1,
    );

    let traces = vec![poseidon2_trace.clone(), dummy_trace.clone()];

    // engine generation
    let perm = random_perm();
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3);
    let engine = engine_from_perm(perm, fri_params);

    // positive test
    engine
        .run_simple_test_impl(
            any_rap_arc_vec![poseidon2_air.clone(), page_requester],
            traces,
            vec![vec![]; 2],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let width = rng.gen_range(0..poseidon2_air.get_width());
        let height = rng.gen_range(0..num_rows);
        let rand = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut(height)[width] += rand;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    any_rap_arc_vec![poseidon2_air.clone(), page_requester],
                    vec![poseidon2_trace.clone(), dummy_trace.clone()],
                    vec![vec![]; 2],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut(height)[width] -= rand;
    }
}

#[test]
fn test_horizen_poseidon2() {
    let horizen_permut = HorizenPoseidon2::new(&POSEIDON2_BABYBEAR_16_PARAMS);
    let mut rng = create_seeded_rng();
    let (external_round_constants, internal_round_constants, horizen_int_diag) =
        Poseidon2Air::<16, BabyBear>::horizen_round_consts_16();
    let mut air_permut = Poseidon2Air::<16, BabyBear>::new(
        external_round_constants,
        internal_round_constants,
        HL_MDS_MAT_4,
        horizen_int_diag,
        BabyBear::ONE,
        3,
        0,
    );
    let u32state = (0..16)
        .map(|_| rng.gen_range(1..=1 << 27))
        .collect::<Vec<_>>();
    let horizen_state: Vec<HorizenBabyBear> =
        u32state.into_iter().map(HorizenBabyBear::from).collect();
    let p3_state: [BabyBear; 16] = horizen_state
        .iter()
        .copied()
        .map(Poseidon2Air::<16, BabyBear>::horizen_to_p3)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let air_result: Vec<BabyBear> = air_permut.request_trace(&[p3_state])[0].clone();
    let horizen_result = horizen_permut.permutation(&horizen_state);
    let air_u32_result = air_result
        .iter()
        .map(BabyBear::as_canonical_u32)
        .collect::<Vec<_>>();
    let horizen_u32_result = horizen_result
        .into_iter()
        .map(|elem| elem.into_bigint().0[0] as u32)
        .collect::<Vec<_>>();
    assert_eq!(air_u32_result, horizen_u32_result);
}
