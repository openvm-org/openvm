use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use stark_backend_v2::{
    DIGEST_SIZE, F, SC, StarkWhirEngine, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::DuplexSpongeRecorder,
    proof::Proof,
    prover::{CommittedTraceDataV2, DeviceDataTransporterV2, ProverBackendV2, ProvingContextV2},
};

use crate::{
    aggregation::{
        AggregationCircuit, AggregationProver, MAX_NUM_PROOFS, trace_heights_tracing_info,
    },
    public_values::AggNodeTraceGen,
};

#[derive(Clone, Debug)]
pub struct NonRootStarkProof {
    pub inner: Proof,
    pub user_pvs_proof: UserPublicValuesProof<DIGEST_SIZE, F>,
}

/*
 * Struct to generate an aggregation proof
 */
pub struct NonRootAggregationProver<
    PB: ProverBackendV2,
    S: AggregationSubCircuit,
    T: AggNodeTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKeyV2>,
    vk: Arc<MultiStarkVerifyingKeyV2>,

    agg_node_tracegen: T,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<PB>,
    circuit: Arc<AggregationCircuit<S>>,
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: AggNodeTraceGen<PB>>
    AggregationProver<PB> for NonRootAggregationProver<PB, S, T>
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2> {
        self.vk.clone()
    }

    fn get_commit(&self) -> PB::Commitment {
        self.child_vk_pcs_data.commitment.clone()
    }

    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> ProvingContextV2<PB> {
        assert!(proofs.len() <= MAX_NUM_PROOFS);
        let verifier_pvs_ctx = self
            .agg_node_tracegen
            .generate_verifier_pvs_ctx(proofs, user_pv_commit);
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proofs, user_pv_commit);
        ProvingContextV2 {
            per_trace: once(verifier_pvs_ctx)
                .chain(
                    self.circuit
                        .verifier_circuit
                        .generate_proving_ctxs::<DuplexSpongeRecorder>(
                            &self.child_vk,
                            self.child_vk_pcs_data.clone(),
                            proofs,
                        ),
                )
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }

    fn agg_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
    ) -> Result<Proof> {
        let ctx = self.generate_proving_ctx(proofs, user_pv_commit);
        if tracing::enabled!(tracing::Level::INFO) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params);
        Ok(engine.prove(
            &engine.device().transport_pk_to_device(self.pk.as_ref()),
            ctx,
        ))
    }
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: AggNodeTraceGen<PB>>
    NonRootAggregationProver<PB, S, T>
{
    pub fn new<E: StarkWhirEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        system_params: SystemParams,
    ) -> Self {
        let verifier_circuit = S::new(child_vk.clone(), true);
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(AggregationCircuit::new(Arc::new(verifier_circuit)));
        let (pk, vk) = engine.keygen(&circuit.airs());
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }

    pub fn get_circuit(&self) -> Arc<AggregationCircuit<S>> {
        self.circuit.clone()
    }
}
