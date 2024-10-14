use std::sync::Arc;

use itertools::izip;
use p3_matrix::dense::DenseMatrix;
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    parizip,
    prover::{
        types::{AirProofInput, CommittedTraceData, Proof, ProofInput, TraceCommitter},
        MultiTraceStarkProver,
    },
    rap::AnyRap,
    utils::AirInfo,
    verifier::{MultiTraceStarkVerifier, VerificationError},
};

/// Data for verifying a Stark proof.
pub struct VerificationData<SC: StarkGenericConfig> {
    pub vk: MultiStarkVerifyingKey<SC>,
    pub proof: Proof<SC>,
}

/// Testing engine
pub trait StarkEngine<SC: StarkGenericConfig> {
    /// Stark config
    fn config(&self) -> &SC;
    /// Creates a new challenger with a deterministic state.
    /// Creating new challenger for prover and verifier separately will result in
    /// them having the same starting state.
    fn new_challenger(&self) -> SC::Challenger;

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<SC> {
        MultiStarkKeygenBuilder::new(self.config())
    }

    fn prover(&self) -> MultiTraceStarkProver<SC> {
        MultiTraceStarkProver::new(self.config())
    }

    fn verifier(&self) -> MultiTraceStarkVerifier<SC> {
        MultiTraceStarkVerifier::new(self.config())
    }

    // TODO[jpw]: the following does not belong in this crate! dev tooling only

    /// Runs a single end-to-end test for a given set of AIRs and traces.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    /// This function should only be used on AIRs where the main trace is **not** partitioned.
    fn run_simple_test_impl(
        &self,
        chips: Vec<Arc<dyn AnyRap<SC>>>,
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
        keygen_builder: &mut MultiStarkKeygenBuilder<'a, SC>,
        air_infos: &'a [AirInfo<SC>],
    ) -> Vec<usize> {
        air_infos
            .iter()
            .map(|air_info| {
                let air = air_info.air.clone();
                assert_eq!(air_info.cached_traces.len(), air.cached_main_widths().len());
                assert_eq!(air_info.common_trace.width, air.common_main_width());
                keygen_builder.add_air(air)
            })
            .collect()
    }

    fn prove(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        air_infos: &[AirInfo<SC>],
        air_ids: Vec<usize>,
    ) -> Proof<SC>
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
        let air_proof_inputs = izip!(air_ids, air_infos)
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
                        air: air_info.air.clone(),
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
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        let mut challenger = self.new_challenger();
        let verifier = self.verifier();
        verifier.verify(&mut challenger, vk, proof)
    }
}
