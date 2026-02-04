use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, PrimeField};
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use stark_backend_v2::{
    DIGEST_SIZE, F, SC, StarkWhirEngine, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::DuplexSpongeRecorder,
    proof::Proof,
    prover::{
        AirProvingContextV2, CommittedTraceDataV2, DeviceDataTransporterV2, ProverBackendV2,
        ProvingContextV2,
    },
};
use tracing::instrument;

use crate::{
    aggregation::{AggregationProver, Circuit, trace_heights_tracing_info},
    circuit::{
        AggNodeTraceGen,
        dag_commit::{DagCommitAir, generate_dag_commit_proving_ctx},
        nonroot::{receiver::UserPvsReceiverAir, verifier::VerifierPvsAir},
    },
};

#[derive(derive_new::new, Clone)]
pub struct CompressionCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for CompressionCircuit<S> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let public_values_bus = self.verifier_circuit.public_values_bus();
        [Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus: self.verifier_circuit.cached_commit_bus(),
        }) as AirRef<BabyBearPoseidon2Config>]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain([
            Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<BabyBearPoseidon2Config>
        ])
        .chain([
            Arc::new(DagCommitAir::new(self.verifier_circuit.dag_commit_bus()))
                as AirRef<BabyBearPoseidon2Config>,
        ])
        .collect_vec()
    }
}

/*
 * Struct to wrap and compress an aggregation Proof by a) producing a specialized
 * recursion circuit with no cached trace and b) using optimal SystemParams for
 * proof size. Note that this should NOT be used recursively.
 */
pub struct CompressionProver<PB: ProverBackendV2, S: AggregationSubCircuit, T: AggNodeTraceGen<PB>>
{
    pk: Arc<MultiStarkProvingKeyV2>,
    vk: Arc<MultiStarkVerifyingKeyV2>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<PB>,
    circuit: Arc<CompressionCircuit<S>>,

    dag_commit_trace: PB::Matrix,
    dag_commit_pvs: [PB::Val; DIGEST_SIZE],
}

impl<PB, S, T> AggregationProver<PB> for CompressionProver<PB, S, T>
where
    PB: ProverBackendV2,
    S: AggregationSubCircuit + VerifierTraceGen<PB>,
    T: AggNodeTraceGen<PB>,
    PB::Matrix: Clone,
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2> {
        self.vk.clone()
    }

    fn get_cached_commit(&self, _is_recursive: bool) -> <PB as ProverBackendV2>::Commitment {
        self.child_vk_pcs_data.commitment.clone()
    }

    #[instrument(name = "trace_gen", skip_all)]
    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        _is_recursive: bool,
    ) -> ProvingContextV2<PB> {
        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            proofs,
            user_pv_commit,
            self.child_vk_pcs_data.commitment.clone(),
        );
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs::<DuplexSpongeRecorder>(
                &self.child_vk,
                self.child_vk_pcs_data.clone(),
                proofs,
            );
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proofs, user_pv_commit);

        let dag_commit_ctx = AirProvingContextV2 {
            cached_mains: vec![],
            common_main: self.dag_commit_trace.clone(),
            public_values: self.dag_commit_pvs.to_vec(),
        };

        ProvingContextV2 {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .chain(once(dag_commit_ctx))
                .enumerate()
                .collect_vec(),
        }
    }

    #[instrument(name = "total_proof", skip_all)]
    fn agg_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        _is_recursive: bool,
    ) -> Result<Proof> {
        let ctx = self.generate_proving_ctx(proofs, user_pv_commit, false);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::aggregation::debug_constraints(&self.circuit, &ctx.per_trace, &engine);
        let proof = engine.prove(
            &engine.device().transport_pk_to_device(self.pk.as_ref()),
            ctx,
        );
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: AggNodeTraceGen<PB>>
    CompressionProver<PB, S, T>
{
    pub fn new<E: StarkWhirEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        system_params: SystemParams,
    ) -> Self
    where
        E::PD: DeviceDataTransporterV2<PB> + Clone,
        PB::Val: Field + PrimeField,
        PB::Matrix: Clone,
    {
        let verifier_circuit = S::new(child_vk.clone(), true, false);
        let engine = E::new(system_params);
        let circuit = Arc::new(CompressionCircuit::new(Arc::new(verifier_circuit)));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let (dag_commit_trace, dag_commit) = generate_dag_commit_proving_ctx(
            (*engine.device()).clone(),
            child_vk_pcs_data.trace.clone(),
        );
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            dag_commit_trace,
            dag_commit_pvs: dag_commit,
        }
    }

    pub fn from_pk<E: StarkWhirEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        pk: Arc<MultiStarkProvingKeyV2>,
    ) -> Self
    where
        E::PD: DeviceDataTransporterV2<PB> + Clone,
        PB::Val: Field + PrimeField,
        PB::Matrix: Clone,
    {
        let verifier_circuit = S::new(child_vk.clone(), true, false);
        let engine = E::new(pk.params.clone());
        let circuit = Arc::new(CompressionCircuit::new(Arc::new(verifier_circuit)));
        let vk = Arc::new(pk.get_vk());
        let (dag_commit_trace, dag_commit) = generate_dag_commit_proving_ctx(
            (*engine.device()).clone(),
            child_vk_pcs_data.trace.clone(),
        );
        Self {
            pk,
            vk,
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            dag_commit_trace,
            dag_commit_pvs: dag_commit,
        }
    }

    pub fn get_circuit(&self) -> Arc<CompressionCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_dag_commit(&self) -> [PB::Val; DIGEST_SIZE] {
        self.dag_commit_pvs
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKeyV2> {
        self.pk.clone()
    }
}
