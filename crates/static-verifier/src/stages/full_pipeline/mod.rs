use halo2_base::{utils::biguint_to_fe, AssignedValue};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config as RootConfig, Bn254Scalar},
    openvm_stark_backend::{
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        p3_field::{PrimeCharacteristicRing, PrimeField},
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
    p3_baby_bear::BabyBear,
};

use crate::{
    chip_traits::{DigestHashInst, PopulateInputs, TranscriptInst},
    field::baby_bear::ReducedBabyBearWire,
    stages::{
        batch_constraints::{
            constrain_batch_constraints_verification, load_batch_constraint_proof_wire,
            load_gkr_proof_wire, BatchConstraintProofWire, GkrProofWire,
        },
        proof_shape::log_heights_per_air_from_proof,
        stacked_reduction::{
            constrain_stacked_reduction, load_stacking_proof_wire, StackingProofWire,
        },
        whir::{constrain_whir_verification, load_whir_proof_wire, WhirProofWire},
    },
    transcript::{digest_wire_from_root, DigestWire},
    Fr,
};

mod public_values;
#[cfg(test)]
mod tests;

pub use public_values::*;

#[derive(Clone, Debug)]
pub struct ProofWire<F = AssignedValue<Fr>> {
    pub common_main_commit_root: F,
    pub public_values: Vec<Vec<ReducedBabyBearWire<F>>>,
    pub cached_commitment_roots: Vec<Vec<F>>,
    pub gkr: GkrProofWire<F>,
    pub batch: BatchConstraintProofWire<F>,
    pub stacking: StackingProofWire<F>,
    pub whir: WhirProofWire<F>,
}

pub(crate) fn digest_scalar_to_fr(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

/// Load proof data into circuit cells. `log_heights_per_air` must match this circuit's fixed
/// heights; host-side asserts that per-AIR log heights extracted from the proof match
/// `log_heights_per_air`.
pub fn load_proof_wire<B: PopulateInputs>(
    b: &mut B,
    proof: &Proof<RootConfig>,
    log_heights_per_air: &[usize],
) -> ProofWire<B::F> {
    let from_proof = log_heights_per_air_from_proof(proof);
    assert_eq!(
        from_proof.as_slice(),
        log_heights_per_air,
        "per-AIR log heights from proof must match this circuit's fixed log_heights_per_air"
    );

    let common_main_commit_root = b.load_witness(digest_scalar_to_fr(proof.common_main_commit[0]));

    let public_values = proof
        .public_values
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| b.bb_load_reduced_witness(value))
                .collect()
        })
        .collect();

    let cached_commitment_roots = proof
        .trace_vdata
        .iter()
        .map(|vdata| {
            if let Some(vdata) = vdata {
                vdata
                    .cached_commitments
                    .iter()
                    .map(|commit| b.load_witness(digest_scalar_to_fr(commit[0])))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let gkr = load_gkr_proof_wire(b, &proof.gkr_proof);
    let batch = load_batch_constraint_proof_wire(b, &proof.batch_constraint_proof);
    let stacking = load_stacking_proof_wire(b, &proof.stacking_proof);
    let whir = load_whir_proof_wire(b, &proof.whir_proof);

    ProofWire {
        common_main_commit_root,
        public_values,
        cached_commitment_roots,
        gkr,
        batch,
        stacking,
        whir,
    }
}

#[allow(clippy::too_many_arguments)]
fn observe_preamble<B: TranscriptInst>(
    b: &mut B,
    mvk: &MultiStarkVerifyingKey<RootConfig>,
    log_heights_per_air: &[usize],
    public_values: &[Vec<ReducedBabyBearWire<B::F>>],
    cached_commitment_roots: &[Vec<B::F>],
    vk_pre_hash: DigestWire<B::F>,
    common_main_commit: DigestWire<B::F>,
) {
    b.observe_commit(&vk_pre_hash);
    b.observe_commit(&common_main_commit);

    for air_idx in 0..mvk.inner.per_air.len() {
        if !mvk.inner.per_air[air_idx].is_required {
            // Static verifier: every AIR in the child VK has a trace (see crate `lib.rs`).
            // unfortunately transcript has it's own cloned BabyBear chip that has a separate
            // constant cache compared to the BabyBear chip. To make vk the same as on main we need
            // this.
            // TODO: if vk ever can change then remove this
            let presence_flag = b.transcript_load_reduced_constant(BabyBear::ONE);
            b.observe(&presence_flag);
        }

        if let Some(preprocessed) = mvk.inner.per_air[air_idx].preprocessed_data.as_ref() {
            let preprocessed_root = b.load_constant(digest_scalar_to_fr(preprocessed.commit[0]));
            b.observe_commit(&digest_wire_from_root(preprocessed_root));
        } else {
            // Fixed circuit parameter (not loaded from the proof witness).
            let lh = u32::try_from(log_heights_per_air[air_idx])
                .expect("log_height must fit in u32 for BabyBear constant");
            let log_height = b.transcript_load_reduced_constant(BabyBear::from_u32(lh));
            b.observe(&log_height);
        }

        for root in &cached_commitment_roots[air_idx] {
            b.observe_commit(&digest_wire_from_root(*root));
        }

        for value in &public_values[air_idx] {
            b.observe(value);
        }
    }
}

/// Run the full static verifier pipeline on pre-loaded witness data.
///
/// `trace_id_to_air_id` and `log_heights_per_air` are fixed for this circuit (host-side). They must
/// match the child proof shape: `log_heights_per_air.len() == mvk.inner.per_air.len()`, and
/// `trace_id_to_air_id` must list every `air_id` exactly once in descending-`log_height` order
/// (tie-break: ascending `air_id`).
///
/// `stacked_layouts` must be the layout vector fixed for this circuit (same as stored on
/// [`crate::StaticVerifierCircuit`]).
pub fn constrained_verify<B: TranscriptInst + DigestHashInst>(
    b: &mut B,
    root_vk: &MultiStarkVerifyingKey<RootConfig>,
    proof_wire: &ProofWire<B::F>, /* Root proof */
    trace_id_to_air_id: &[usize],
    log_heights_per_air: &[usize],
    stacked_layouts: &[StackedLayout],
) {
    assert_eq!(
        log_heights_per_air.len(),
        root_vk.inner.per_air.len(),
        "log_heights_per_air must match VK per_air count"
    );
    let l_skip = root_vk.inner.params.l_skip;
    let n_per_trace: Vec<isize> = trace_id_to_air_id
        .iter()
        .map(|&air_id| log_heights_per_air[air_id] as isize - l_skip as isize)
        .collect();

    let mut profiler = crate::profiling::CellProfiler::new("constrained_verify", b.cell_count());

    let mvk_pre_hash_root = b.load_constant(digest_scalar_to_fr(root_vk.pre_hash[0]));
    b.init_transcript();

    profiler.push("observe_preamble", b.cell_count());
    observe_preamble(
        b,
        root_vk,
        log_heights_per_air,
        &proof_wire.public_values,
        &proof_wire.cached_commitment_roots,
        digest_wire_from_root(mvk_pre_hash_root),
        digest_wire_from_root(proof_wire.common_main_commit_root),
    );
    profiler.pop(b.cell_count());

    profiler.push("batch_constraints", b.cell_count());
    let batch = constrain_batch_constraints_verification(
        b,
        &root_vk.inner,
        &proof_wire.gkr,
        &proof_wire.batch,
        &n_per_trace,
        trace_id_to_air_id,
        proof_wire.public_values.clone(),
        &mut profiler,
    );
    profiler.pop(b.cell_count());

    let need_rot_per_commit = get_need_rot_per_commit(&root_vk.inner, trace_id_to_air_id);

    profiler.push("stacked_reduction", b.cell_count());
    let stacked_reduction = constrain_stacked_reduction(
        b,
        &proof_wire.stacking,
        stacked_layouts,
        &need_rot_per_commit,
        l_skip,
        root_vk.inner.params.n_stack,
        &batch.column_openings,
        &batch.r,
        &mut profiler,
    );
    profiler.pop(b.cell_count());

    let u_cube = {
        let u = &stacked_reduction.u;
        assert!(!u.is_empty());
        let mut u_cube = Vec::with_capacity(l_skip + u.len().saturating_sub(1));
        let mut power = *u.first().unwrap();
        for _ in 0..l_skip {
            u_cube.push(power);
            power = b.ext_square(power);
            power = b.ext_reduce_max_bits(power);
        }
        u_cube.extend(u.iter().skip(1).copied());
        u_cube
    };

    let initial_commitment_roots = {
        let common_main_root = proof_wire.common_main_commit_root;
        let mut commits = vec![common_main_root];
        for &air_id in trace_id_to_air_id {
            if let Some(preprocessed) = &root_vk.inner.per_air[air_id].preprocessed_data {
                commits.push(b.load_constant(digest_scalar_to_fr(preprocessed.commit[0])));
            }
            commits.extend(proof_wire.cached_commitment_roots[air_id].iter().copied());
        }
        commits
    };

    profiler.push("whir_verification", b.cell_count());
    constrain_whir_verification(
        b,
        &root_vk.inner,
        &proof_wire.whir,
        &stacked_reduction.stacking_openings,
        &initial_commitment_roots,
        &u_cube,
        &mut profiler,
    );
    profiler.pop(b.cell_count());

    profiler.print(b.cell_count());

    #[cfg(feature = "cell-profiling")]
    if let Ok(dir) = std::env::var("OPENVM_PROFILE_DIR") {
        let _ = std::fs::create_dir_all(&dir);
        profiler.write_flamegraph(
            &format!("{dir}/constrained_verify.svg"),
            "Constrained Verify Sub-stages",
            b.cell_count(),
        );
        profiler.write_flamegraph_reversed(
            &format!("{dir}/constrained_verify_rev.svg"),
            "Constrained Verify Sub-stages (reversed)",
            b.cell_count(),
        );
    }
}

/// Helper function, purely on out-of-circuit values.
fn get_need_rot_per_commit(
    mvk0: &MultiStarkVerifyingKey0<RootConfig>,
    trace_id_to_air_id: &[usize],
) -> Vec<Vec<bool>> {
    let mut need_rot_per_commit = vec![trace_id_to_air_id
        .iter()
        .map(|&air_id| mvk0.per_air[air_id].params.need_rot)
        .collect::<Vec<_>>()];
    for &air_id in trace_id_to_air_id {
        let need_rot = mvk0.per_air[air_id].params.need_rot;
        if mvk0.per_air[air_id].preprocessed_data.is_some() {
            need_rot_per_commit.push(vec![need_rot]);
        }
        let cached_len = mvk0.per_air[air_id].params.width.cached_mains.len();
        for _ in 0..cached_len {
            need_rot_per_commit.push(vec![need_rot]);
        }
    }
    need_rot_per_commit
}
