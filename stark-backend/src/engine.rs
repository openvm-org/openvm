use itertools::izip;
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};

use crate::{
    commit::SingleMatrixCommitPtr,
    config::{Com, PcsProof, PcsProverData},
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{trace::TraceCommitmentBuilder, types::Proof, MultiTraceStarkProver},
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

    fn trace_commitment_builder<'a>(&'a self) -> TraceCommitmentBuilder<'a, SC>
    where
        SC: 'a,
    {
        TraceCommitmentBuilder::new(self.config().pcs())
    }

    fn prover(&self) -> MultiTraceStarkProver<SC> {
        MultiTraceStarkProver::new(self.config())
    }

    fn verifier(&self) -> MultiTraceStarkVerifier<SC> {
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
        self.set_up_keygen_builder(&mut keygen_builder, air_infos);
        let pk = keygen_builder.generate_pk();
        let vk = pk.vk();
        let proof = self.prove(&pk, air_infos);
        self.verify(&vk, &proof)?;
        Ok(VerificationData { vk, proof })
    }

    fn set_up_keygen_builder<'a>(
        &self,
        keygen_builder: &mut MultiStarkKeygenBuilder<'a, SC>,
        air_infos: &'a [AirInfo<SC>],
    ) {
        for air_info in air_infos {
            // Note: we count the common main trace always even when its width is 0
            assert_eq!(
                air_info.air.cached_main_widths().len(),
                air_info.cached_traces.len()
            );
        }

        let mut trace_ptrs: Vec<Vec<SingleMatrixCommitPtr>> = vec![vec![]; air_infos.len()];
        // First, create pointers for cached traces
        for (air_info, air_trace_ptrs) in izip!(air_infos, trace_ptrs.iter_mut()) {
            let cached_trace_widths = air_info.air.cached_main_widths();
            for width in cached_trace_widths {
                air_trace_ptrs.push(keygen_builder.add_cached_main_matrix(width));
            }
        }
        // Second, create pointer for common traces
        for (air_info, air_trace_ptrs) in izip!(air_infos, trace_ptrs.iter_mut()) {
            let common_trace_width = air_info.air.common_main_width();
            air_trace_ptrs.push(keygen_builder.add_main_matrix(common_trace_width));
        }
        // Third, register all AIRs using the trace pointers
        for (air_info, air_trace_ptrs) in izip!(air_infos, trace_ptrs) {
            keygen_builder.add_partitioned_air(air_info.air.as_ref(), air_trace_ptrs);
        }
    }

    fn prove(&self, pk: &MultiStarkProvingKey<SC>, air_infos: &[AirInfo<SC>]) -> Proof<SC>
    where
        SC::Pcs: Sync,
        Domain<SC>: Send + Sync,
        PcsProverData<SC>: Send + Sync,
        Com<SC>: Send + Sync,
        SC::Challenge: Send + Sync,
        PcsProof<SC>: Send + Sync,
    {
        let vk = pk.vk();

        let prover = self.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        // First, load all cached traces
        for air_info in air_infos {
            let cached_traces = air_info.cached_traces.clone();
            for cached_trace in cached_traces {
                let pdata = trace_builder.committer.commit(vec![cached_trace.clone()]);
                trace_builder.load_cached_trace(cached_trace, pdata);
            }
        }
        // Second, load the single common trace for each chip
        for air_info in air_infos {
            trace_builder.load_trace(air_info.common_trace.clone());
        }

        trace_builder.commit_current();

        let main_trace_data = trace_builder.view(
            &vk,
            air_infos
                .iter()
                .map(|air_info| air_info.air.as_ref())
                .collect(),
        );

        let mut challenger = self.new_challenger();

        #[cfg(feature = "bench-metrics")]
        let prove_start = std::time::Instant::now();

        let public_values: Vec<Vec<Val<SC>>> = air_infos
            .iter()
            .map(|air_info| air_info.public_values.clone())
            .collect();

        prover.prove(&mut challenger, pk, main_trace_data, &public_values)
    }

    fn verify(
        &self,
        vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
    ) -> Result<(), VerificationError> {
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("stark_prove_excluding_trace_time_ms")
            .set(prove_start.elapsed().as_millis() as f64);

        let mut challenger = self.new_challenger();
        let verifier = self.verifier();
        verifier.verify(&mut challenger, vk, proof)
    }
}
