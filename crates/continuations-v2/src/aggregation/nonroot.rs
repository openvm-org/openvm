use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use stark_backend_v2::{
    DIGEST_SIZE, F, SC, StarkWhirEngine, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::DuplexSpongeRecorder,
    proof::Proof,
    prover::{
        CommittedTraceDataV2, DeviceDataTransporterV2, DeviceMultiStarkProvingKeyV2,
        ProverBackendV2, ProvingContextV2,
    },
};
use tracing::instrument;

use crate::{
    aggregation::{AggregationProver, Circuit, trace_heights_tracing_info},
    circuit::{
        AggNodeTraceGen,
        nonroot::{receiver::UserPvsReceiverAir, verifier::VerifierPvsAir},
    },
};

#[derive(derive_new::new, Clone)]
pub struct NonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for NonRootCircuit<S> {
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
        .collect_vec()
    }
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
    d_pk: DeviceMultiStarkProvingKeyV2<PB>,
    vk: Arc<MultiStarkVerifyingKeyV2>,

    agg_node_tracegen: T,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<PB>,
    circuit: Arc<NonRootCircuit<S>>,

    self_vk_pcs_data: Option<CommittedTraceDataV2<PB>>,
}

impl<PB, S, T> AggregationProver<PB> for NonRootAggregationProver<PB, S, T>
where
    PB: ProverBackendV2,
    S: AggregationSubCircuit + VerifierTraceGen<PB>,
    T: AggNodeTraceGen<PB>,
    PB::Matrix: Clone,
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2> {
        self.vk.clone()
    }

    fn get_cached_commit(&self, is_recursive: bool) -> PB::Commitment {
        if is_recursive {
            self.self_vk_pcs_data.as_ref().unwrap().commitment.clone()
        } else {
            self.child_vk_pcs_data.commitment.clone()
        }
    }

    #[instrument(name = "trace_gen", skip_all)]
    fn generate_proving_ctx(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        is_recursive: bool,
    ) -> ProvingContextV2<PB> {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());
        let (child_vk, child_vk_commit) = if is_recursive {
            (&self.vk, self.self_vk_pcs_data.clone().unwrap())
        } else {
            (&self.child_vk, self.child_vk_pcs_data.clone())
        };
        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            proofs,
            user_pv_commit,
            child_vk_commit.commitment.clone(),
        );
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs::<DuplexSpongeRecorder>(
                child_vk,
                child_vk_commit.clone(),
                proofs,
            );
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proofs, user_pv_commit);

        ProvingContextV2 {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }

    #[instrument(name = "total_proof", skip_all)]
    fn agg_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proofs: &[Proof],
        user_pv_commit: Option<[F; DIGEST_SIZE]>,
        is_recursive: bool,
    ) -> Result<Proof> {
        let ctx = self.generate_proving_ctx(proofs, user_pv_commit, is_recursive);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::aggregation::debug_constraints(&self.circuit, &ctx.per_trace, &engine);
        let proof = engine.prove(&self.d_pk, ctx);
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: AggNodeTraceGen<PB>>
    NonRootAggregationProver<PB, S, T>
{
    pub fn new<E: StarkWhirEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        system_params: SystemParams,
        is_recursive: bool,
    ) -> Self {
        let verifier_circuit = S::new(child_vk.clone(), true, true);
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(NonRootCircuit::new(Arc::new(verifier_circuit)));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn from_pk<E: StarkWhirEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        pk: Arc<MultiStarkProvingKeyV2>,
        is_recursive: bool,
    ) -> Self {
        let verifier_circuit = S::new(child_vk.clone(), true, true);
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data: CommittedTraceDataV2<PB> =
            verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(NonRootCircuit::new(Arc::new(verifier_circuit)));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn get_circuit(&self) -> Arc<NonRootCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKeyV2> {
        self.pk.clone()
    }

    pub fn get_self_vk_pcs_data(&self) -> Option<CommittedTraceDataV2<PB>>
    where
        CommittedTraceDataV2<PB>: Clone,
    {
        self.self_vk_pcs_data.clone()
    }
}
