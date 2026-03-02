use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{DeferralPvs, VerifierBasePvs};

use crate::{
    circuit::deferral::{
        aggregation::hook::verifier::air::DeferralHookPvsCols, DeferralAggregationPvs,
        DEF_AGG_PVS_AIR_ID, DEF_AGG_VERIFIER_AIR_ID,
    },
    utils::{digests_to_poseidon2_input, pad_slice_to_poseidon2_input, zero_hash},
    SC,
};

pub fn def_vk_commit_from_verifier_pvs(
    verifier_pvs: &VerifierBasePvs<F>,
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    // Here app_dag_commit is the def_dag_commit
    let intermediate_vk_commit =
        poseidon2_compress_with_capacity(verifier_pvs.app_dag_commit, verifier_pvs.leaf_dag_commit)
            .0;
    let def_vk_commit = poseidon2_compress_with_capacity(
        intermediate_vk_commit,
        verifier_pvs.internal_for_leaf_dag_commit,
    )
    .0;
    (intermediate_vk_commit, def_vk_commit)
}

pub fn generate_proving_ctx(
    proof: &Proof<SC>,
    input_onion: [F; DIGEST_SIZE],
    output_onion: [F; DIGEST_SIZE],
) -> (
    AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    Vec<[F; POSEIDON2_WIDTH]>,
    [F; DIGEST_SIZE],
) {
    let verifier_pvs: &VerifierBasePvs<F> = proof.public_values[DEF_AGG_VERIFIER_AIR_ID]
        .as_slice()
        .borrow();
    let def_pvs: &DeferralAggregationPvs<F> =
        proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();

    let (intermediate_vk_commit, def_vk_commit) = def_vk_commit_from_verifier_pvs(verifier_pvs);

    let width = DeferralHookPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut DeferralHookPvsCols<F> = trace.as_mut_slice().borrow_mut();
    cols.verifier_pvs = *verifier_pvs;
    cols.def_pvs = *def_pvs;
    cols.intermediate_vk_commit = intermediate_vk_commit;
    cols.def_vk_commit = def_vk_commit;
    cols.input_onion = input_onion;
    cols.output_onion = output_onion;

    const ZERO_DIGEST: [F; DIGEST_SIZE] = [F::ZERO; DIGEST_SIZE];
    let def_vk_commit_padded = poseidon2_compress_with_capacity(def_vk_commit, ZERO_DIGEST).0;
    let input_onion_padded = poseidon2_compress_with_capacity(input_onion, ZERO_DIGEST).0;
    let output_onion_padded = poseidon2_compress_with_capacity(output_onion, ZERO_DIGEST).0;
    cols.def_vk_commit_padded = def_vk_commit_padded;
    cols.input_onion_padded = input_onion_padded;
    cols.output_onion_padded = output_onion_padded;

    let zero_hash = zero_hash(1);
    let initial_acc_hash = poseidon2_compress_with_capacity(def_vk_commit_padded, zero_hash).0;
    let final_acc_hash =
        poseidon2_compress_with_capacity(input_onion_padded, output_onion_padded).0;

    let mut public_values = vec![F::ZERO; DeferralPvs::<u8>::width()];
    let root_pvs: &mut DeferralPvs<F> = public_values.as_mut_slice().borrow_mut();
    root_pvs.initial_acc_hash = initial_acc_hash;
    root_pvs.final_acc_hash = final_acc_hash;
    root_pvs.depth = F::ONE;

    let poseidon2_inputs = vec![
        digests_to_poseidon2_input(verifier_pvs.app_dag_commit, verifier_pvs.leaf_dag_commit),
        digests_to_poseidon2_input(
            intermediate_vk_commit,
            verifier_pvs.internal_for_leaf_dag_commit,
        ),
        pad_slice_to_poseidon2_input(&def_vk_commit, F::ZERO),
        pad_slice_to_poseidon2_input(&input_onion, F::ZERO),
        pad_slice_to_poseidon2_input(&output_onion, F::ZERO),
        digests_to_poseidon2_input(def_vk_commit_padded, zero_hash),
        digests_to_poseidon2_input(input_onion_padded, output_onion_padded),
    ];

    (
        AirProvingContext {
            cached_mains: vec![],
            common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
            public_values,
        },
        poseidon2_inputs,
        def_vk_commit,
    )
}
