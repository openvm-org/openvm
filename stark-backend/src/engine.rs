use itertools::izip;
use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        v2::{types::MultiStarkVerifyingKeyV2, MultiStarkKeygenBuilderV2},
        MultiStarkKeygenBuilder,
    },
    parizip,
    prover::{
        trace::{TraceCommitmentBuilder, TraceCommitter},
        v2::{
            types::{AirProofInput, CommittedTraceData, ProofInput, ProofV2},
            MultiTraceStarkProverV2,
        },
        MultiTraceStarkProver,
    },
    rap::AnyRap,
    verifier::{v2::MultiTraceStarkVerifierV2, MultiTraceStarkVerifier, VerificationError},
};

/// Data for verifying a Stark proof.
pub struct VerificationData<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKeyV2<SC>,
    pub proof: ProofV2<SC>,
}

/// Testing engine
pub trait StarkEngine<SC: StarkGenericConfig> {
    /// Stark config
    fn config(&self) -> &SC;
    /// Creates a new challenger with a deterministic state.
    /// Creating new challenger for prover and verifier separately will result in
    /// them having the same starting state.
    fn new_challenger(&self) -> SC::Challenger;

    fn keygen_builder(&self) -> MultiStarkKeygenBuilderV2<SC> {
        MultiStarkKeygenBuilderV2::new(self.config())
    }

    fn keygen_builder_v1(&self) -> MultiStarkKeygenBuilder<SC> {
        MultiStarkKeygenBuilder::new(self.config())
    }

    fn trace_commitment_builder<'a>(&'a self) -> TraceCommitmentBuilder<'a, SC>
    where
        SC: 'a,
    {
        TraceCommitmentBuilder::new(self.config().pcs())
    }

    fn prover(&self) -> MultiTraceStarkProverV2<SC> {
        MultiTraceStarkProverV2::new(self.config())
    }

    fn prover_v1(&self) -> MultiTraceStarkProver<SC> {
        MultiTraceStarkProver::new(self.config())
    }

    fn verifier(&self) -> MultiTraceStarkVerifierV2<SC> {
        MultiTraceStarkVerifierV2::new(self.config())
    }

    fn verifier_v1(&self) -> MultiTraceStarkVerifier<SC> {
        MultiTraceStarkVerifier::new(self.config())
    }

    /// Runs a single end-to-end test for a given set of chips and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on chips where the main trace is **not** partitioned.
    ///
    /// - `chips`, `traces`, `public_values` should be zipped.
    fn run_simple_test(
        &self,
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_test_impl(
            self,
            chips,
            traces.into_iter().map(|t| vec![t]).collect(),
            public_values,
        )
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    ///
    /// - `chips`, `traces`, `public_values` should be zipped.
    fn run_test(
        &self,
        chips: &[&dyn AnyRap<SC>],
        traces: Vec<Vec<DenseMatrix<Val<SC>>>>,
        public_values: &[Vec<Val<SC>>],
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        run_test_impl(self, chips, traces, public_values)
    }
}

fn run_test_impl<SC: StarkGenericConfig, E: StarkEngine<SC> + ?Sized>(
    engine: &E,
    chips: &[&dyn AnyRap<SC>],
    traces: Vec<Vec<DenseMatrix<Val<SC>>>>,
    public_values: &[Vec<Val<SC>>],
) -> Result<VerificationData<SC>, VerificationError>
where
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    assert_eq!(chips.len(), traces.len());
    let mut keygen_builder = engine.keygen_builder();
    let air_ids: Vec<_> = izip!(chips, &traces)
        .map(|(chip, chip_traces)| {
            // Note: we count the common main trace always even when its width is 0
            let mut num_traces = chip.cached_main_widths().len();
            if chip.common_main_width() > 0 {
                num_traces += 1;
            }
            assert_eq!(chip_traces.len(), num_traces);
            keygen_builder.add_air(*chip)
        })
        .collect();

    let pk = keygen_builder.generate_pk();

    let commiter = TraceCommitter::new(engine.config().pcs());
    let air_proof_inputs = parizip!(air_ids, chips, traces, public_values.to_vec())
        .map(|(air_id, chip, mut chip_traces, public_values)| {
            let common_main = if chip.common_main_width() > 0 {
                chip_traces.pop()
            } else {
                None
            };
            let cached_mains = parizip!(chip_traces)
                .map(|trace| CommittedTraceData {
                    raw_data: trace.clone(),
                    prover_data: commiter.commit(vec![trace]),
                })
                .collect();
            (
                air_id,
                AirProofInput {
                    air: *chip,
                    cached_mains,
                    common_main,
                    public_values,
                },
            )
        })
        .collect();
    let proof_input = ProofInput {
        per_air: air_proof_inputs,
    };

    let mut challenger = engine.new_challenger();

    #[cfg(feature = "bench-metrics")]
    let prove_start = std::time::Instant::now();

    let prover = engine.prover();
    let proof = prover.prove(&mut challenger, &pk, proof_input);

    #[cfg(feature = "bench-metrics")]
    metrics::gauge!("stark_prove_excluding_trace_time_ms")
        .set(prove_start.elapsed().as_millis() as f64);

    let mut challenger = engine.new_challenger();
    let verifier = engine.verifier();
    verifier.verify(&mut challenger, &pk.get_vk(), &proof)?;
    Ok(VerificationData {
        vk: pk.get_vk(),
        proof,
    })
}
