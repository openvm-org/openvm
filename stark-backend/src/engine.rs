use std::sync::Arc;

use itertools::izip;
use p3_matrix::dense::DenseMatrix;

use crate::{
    config::{StarkGenericConfig, Val},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{
        types::{AirProofInput, Proof, ProofInput, TraceCommitter},
        MultiTraceStarkProver,
    },
    rap::AnyRap,
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
    ) -> Result<VerificationData<SC>, VerificationError> {
        self.run_test_impl(AirProofInput::multiple_simple(chips, traces, public_values))
    }

    /// Runs a single end-to-end test for a given set of chips and traces partitions.
    /// This includes proving/verifying key generation, creating a proof, and verifying the proof.
    fn run_test_impl(
        &self,
        air_proof_inputs: Vec<AirProofInput<SC>>,
    ) -> Result<VerificationData<SC>, VerificationError> {
        let mut keygen_builder = self.keygen_builder();
        let air_ids = self.set_up_keygen_builder(&mut keygen_builder, &air_proof_inputs);
        let proof_input = ProofInput {
            per_air: izip!(air_ids, air_proof_inputs).collect(),
        };
        let pk = keygen_builder.generate_pk();
        let vk = pk.get_vk();
        let proof = self.prove(&pk, proof_input);
        self.verify(&vk, &proof)?;
        Ok(VerificationData { vk, proof })
    }

    /// Add AIRs and get AIR IDs
    fn set_up_keygen_builder(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'_, SC>,
        air_proof_inputs: &[AirProofInput<SC>],
    ) -> Vec<usize> {
        air_proof_inputs
            .iter()
            .map(|air_proof_input| {
                let air = air_proof_input.air.clone();
                assert_eq!(
                    air_proof_input.raw.cached_mains.len(),
                    air.cached_main_widths().len()
                );
                let common_main_width = air.common_main_width();
                if common_main_width == 0 {
                    assert!(air_proof_input.raw.common_main.is_none());
                } else {
                    assert_eq!(
                        air_proof_input.raw.common_main.as_ref().unwrap().width,
                        air.common_main_width()
                    );
                }
                keygen_builder.add_air(air)
            })
            .collect()
    }

    fn prove_then_verify(
        &self,
        pk: &MultiStarkProvingKey<SC>,
        proof_input: ProofInput<SC>,
    ) -> Result<(), VerificationError> {
        let proof = self.prove(pk, proof_input);
        self.verify(&pk.get_vk(), &proof)
    }

    fn prove(&self, pk: &MultiStarkProvingKey<SC>, proof_input: ProofInput<SC>) -> Proof<SC> {
        let prover = self.prover();
        let committer = TraceCommitter::new(prover.pcs());

        let air_proof_inputs = proof_input
            .per_air
            .into_iter()
            .map(|(air_id, mut air_proof_input)| {
                // Commit cached traces if they are not provided
                if air_proof_input.cached_mains_pdata.is_empty()
                    && !air_proof_input.raw.cached_mains.is_empty()
                {
                    air_proof_input.cached_mains_pdata = air_proof_input
                        .raw
                        .cached_mains
                        .iter()
                        .map(|trace| committer.commit(vec![trace.as_ref().clone()]))
                        .collect();
                } else {
                    assert_eq!(
                        air_proof_input.cached_mains_pdata.len(),
                        air_proof_input.raw.cached_mains.len()
                    );
                }
                (air_id, air_proof_input)
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
