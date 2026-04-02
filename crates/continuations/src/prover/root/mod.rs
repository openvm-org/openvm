use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_recursion_circuit::{
    batch_constraint::expr_eval::CachedTraceRecord,
    system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen},
};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{DeviceDataTransporter, ProverBackend, ProvingContext},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_bn254::Bn254;
use p3_field::{Field, PrimeField32};
use tracing::instrument;

use crate::{
    circuit::{
        root::{RootCircuit, RootTraceGen},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    CommitBytes, RootSC, VkCommitBytes, SC,
};

mod trace;

pub struct RootProver<S: AggregationSubCircuit, T> {
    pk: Arc<MultiStarkProvingKey<RootSC>>,
    vk: Arc<MultiStarkVerifyingKey<RootSC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    cached_trace_record: CachedTraceRecord,
    circuit: Arc<RootCircuit<S>>,
    trace_heights: Option<Vec<usize>>,
}

impl<S: AggregationSubCircuit, T> RootProver<S, T> {
    #[instrument(name = "total_proof", skip_all)]
    pub fn root_prove_from_ctx<E>(&self, ctx: ProvingContext<E::PB>) -> Result<Proof<RootSC>>
    where
        E: StarkEngine<SC = RootSC>,
        E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = [Bn254; 1]>,
        <E::PB as ProverBackend>::Matrix: Clone,
        S: VerifierTraceGen<E::PB, RootSC>,
        T: RootTraceGen<E::PB>,
    {
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, RootSC>(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        }
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx)?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }
}

impl<S: AggregationSubCircuit, T> RootProver<S, T> {
    pub fn new<E>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<CommitBytes>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self
    where
        E: StarkEngine<SC = RootSC>,
        E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = [Bn254; 1]>,
        S: VerifierTraceGen<E::PB, RootSC>,
        T: RootTraceGen<E::PB>,
        E::PD: DeviceDataTransporter<RootSC, E::PB> + Clone,
        <E::PB as ProverBackend>::Val: Field + PrimeField32,
        <E::PB as ProverBackend>::Matrix: Clone,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: false,
                ..Default::default()
            },
        );
        let cached_trace_record = verifier_circuit.cached_trace_record(&child_vk);
        let engine = E::new(system_params);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(def_hook_commit.is_some()),
            child_vk,
            cached_trace_record,
            circuit,
            trace_heights,
        }
    }

    pub fn from_pk<E>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<RootSC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<CommitBytes>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self
    where
        E: StarkEngine<SC = RootSC>,
        E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = [Bn254; 1]>,
        S: VerifierTraceGen<E::PB, RootSC>,
        T: RootTraceGen<E::PB>,
        <E::PB as ProverBackend>::Val: Field + PrimeField32,
        <E::PB as ProverBackend>::Matrix: Clone,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: false,
                ..Default::default()
            },
        );
        let cached_trace_record = verifier_circuit.cached_trace_record(&child_vk);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let vk = Arc::new(pk.get_vk());
        Self {
            pk,
            vk,
            agg_node_tracegen: T::new(def_hook_commit.is_some()),
            child_vk,
            cached_trace_record,
            circuit,
            trace_heights,
        }
    }

    pub fn get_circuit(&self) -> Arc<RootCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<RootSC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<RootSC>> {
        self.vk.clone()
    }

    pub fn get_trace_heights(&self) -> Option<Vec<usize>> {
        self.trace_heights.clone()
    }
}
