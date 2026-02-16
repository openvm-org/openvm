use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_poseidon2_air::BABY_BEAR_POSEIDON2_SBOX_DEGREE;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        AirProvingContext, CommittedTraceData, DeviceDataTransporter, ProverBackend, ProvingContext,
    },
    AirRef, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, EF, F,
};
use p3_field::{Field, InjectiveMonomial, PrimeField};
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use tracing::instrument;

use crate::{
    aggregation::{trace_heights_tracing_info, Circuit},
    circuit::{
        dag_commit::{generate_dag_commit_proving_ctx, DagCommitAir},
        nonroot::{receiver::UserPvsReceiverAir, verifier::VerifierPvsAir, NonRootTraceGen},
    },
    SC,
};

#[derive(derive_new::new, Clone)]
pub struct CompressionCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for CompressionCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let public_values_bus = bus_inventory.public_values_bus;
        [Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
        }) as AirRef<SC>]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain([Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<SC>])
        .chain([Arc::new(DagCommitAir::new(bus_inventory.dag_commit_bus)) as AirRef<SC>])
        .collect_vec()
    }
}

/// Wraps and compresses an aggregation Proof by a) producing a specialized
/// recursion circuit with no cached trace and b) using optimal SystemParams for
/// proof size. Note that this should NOT be used recursively.
pub struct CompressionProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: NonRootTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<CompressionCircuit<S>>,

    dag_commit_trace: PB::Matrix,
    dag_commit_pvs: [PB::Val; DIGEST_SIZE],
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > CompressionProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(&self, proof: Proof<SC>) -> ProvingContext<PB> {
        let proof_slice = &[proof];
        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            proof_slice,
            false,
            self.child_vk_pcs_data.commitment.clone(),
        );
        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs_base(
            &self.child_vk,
            self.child_vk_pcs_data.clone(),
            proof_slice,
            default_duplex_sponge_recorder(),
        );
        let agg_other_ctxs = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(proof_slice, false);

        let dag_commit_ctx = AirProvingContext {
            cached_mains: vec![],
            common_main: self.dag_commit_trace.clone(),
            public_values: self.dag_commit_pvs.to_vec(),
        };

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .chain(once(dag_commit_ctx))
                .enumerate()
                .collect_vec(),
        }
    }

    #[instrument(name = "total_proof", skip_all)]
    pub fn compress_prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proof);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::aggregation::debug_constraints(&self.circuit, &ctx, &engine);
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx);
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > CompressionProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        system_params: SystemParams,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, PB> + Clone,
        PB::Val: Field + PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
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

    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, PB> + Clone,
        PB::Val: Field + PrimeField + InjectiveMonomial<BABY_BEAR_POSEIDON2_SBOX_DEGREE>,
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

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    pub fn get_cached_commit(&self) -> <PB as ProverBackend>::Commitment {
        self.child_vk_pcs_data.commitment.clone()
    }
}
