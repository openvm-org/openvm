use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        v2::{
            types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
            MultiStarkKeygenBuilderV2,
        },
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
    utils::AirInfo,
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

    /// Runs a single end-to-end test for a given set of AIRs and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on AIRs where the main trace is **not** partitioned.
    fn run_simple_test_impl(
        &self,
        chips: Vec<Box<dyn AnyRap<SC>>>,
        traces: Vec<DenseMatrix<Val<SC>>>,
        public_values: Vec<Vec<Val<SC>>>,
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        self.run_test_impl(&AirInfo::multiple_simple(chips, traces, public_values))
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test_impl(
        &self,
        air_infos: &[AirInfo<SC>],
    ) -> Result<VerificationData<SC>, VerificationError>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let mut keygen_builder = self.keygen_builder();
        let air_ids = self.set_up_keygen_builder(&mut keygen_builder, air_infos);
        let pk = keygen_builder.generate_pk();
        let vk = pk.get_vk();
        let proof = self.prove(&pk, air_infos, air_ids);
        self.verify(&vk, &proof)?;
        Ok(VerificationData { vk, proof })
    }

    /// Add AIRs and get AIR IDs
    fn set_up_keygen_builder<'a>(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilderV2<'a, SC>,
        air_infos: &'a [AirInfo<SC>],
    ) -> Vec<usize> {
        air_infos
            .iter()
            .map(|air_info| {
                let air = &air_info.air;
                assert_eq!(air_info.cached_traces.len(), air.cached_main_widths().len());
                assert_eq!(air_info.common_trace.width, air.common_main_width());
                keygen_builder.add_air(air.as_ref())
            })
            .collect()
    }

    fn prove(
        &self,
        pk: &MultiStarkProvingKeyV2<SC>,
        air_infos: &[AirInfo<SC>],
        air_ids: Vec<usize>,
    ) -> ProofV2<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let prover = self.prover();
        let committer = TraceCommitter::new(prover.pcs());

        // Commit to the cached traces
        let air_proof_inputs = parizip!(air_ids, air_infos)
            .map(|(air_id, air_info)| {
                let cached_mains = parizip!(air_info.cached_traces.clone())
                    .map(|trace| CommittedTraceData {
                        raw_data: trace.clone(),
                        prover_data: committer.commit(vec![trace]),
                    })
                    .collect();
                (
                    air_id,
                    AirProofInput {
                        air: air_info.air.as_ref(),
                        cached_mains,
                        common_main: Some(air_info.common_trace.clone()),
                        public_values: air_info.public_values.clone(),
                    },
                )
            })
            .collect();
        let proof_input = ProofInput {
            per_air: air_proof_inputs,
        };

        let mut challenger = self.new_challenger();

        #[cfg(feature = "bench-metrics")]
        let prove_start = std::time::Instant::now();
        let _proof = prover.prove(&mut challenger, pk, proof_input);
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("stark_prove_excluding_trace_time_ms")
            .set(prove_start.elapsed().as_millis() as f64);

        _proof
    }

    fn verify(
        &self,
        vk: &MultiStarkVerifyingKeyV2<SC>,
        proof: &ProofV2<SC>,
    ) -> Result<(), VerificationError> {
        let mut challenger = self.new_challenger();
        let verifier = self.verifier();
        verifier.verify(&mut challenger, vk, proof)
    }
}
