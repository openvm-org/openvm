use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, Executor,
        MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig,
    },
    system::memory::merkle::MerkleTree,
};
use openvm_continuations::circuit::inner::ProofsType;
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, Val};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use openvm_verify_stark_host::{
    pvs::{DeferralPvs, DEF_PVS_AIR_ID},
    vk::VerificationBaseline,
    NonRootStarkProof,
};

use crate::{
    prover::{
        deferral::compute_deferral_merkle_proofs, vm::types::VmProvingKey, AggProver, AppProver,
        DeferralProver,
    },
    DeferralInput, StdIn, SC,
};

pub struct StarkProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
    pub def_prover: Option<Arc<DeferralPathProver>>,
}

#[derive(derive_new::new)]
pub struct DeferralPathProver {
    pub deferral_prover: Arc<DeferralProver>,
    pub agg_prover: Arc<AggProver>,
}

impl<E, VB> StarkProver<E, VB>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    Val<SC>: PrimeField32,
{
    pub fn new(
        vm_builder: VB,
        app_vm_pk: &VmProvingKey<VB::VmConfig>,
        app_exe: Arc<VmExe<Val<SC>>>,
        agg_prover: Arc<AggProver>,
        def_prover: Option<Arc<DeferralPathProver>>,
    ) -> Result<Self> {
        Ok(Self {
            app_prover: AppProver::new(vm_builder, app_vm_pk, app_exe)?,
            agg_prover,
            def_prover,
        })
    }

    pub fn prove(
        &mut self,
        vm_input: StdIn<Val<SC>>,
        def_inputs: &[DeferralInput],
    ) -> Result<NonRootStarkProof>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let has_deferrals = self.def_prover.is_some();
        let memory_dimensions = self.app_prover.memory_dimensions();

        // Build the initial memory merkle tree before proving (needed for deferral proofs).
        let initial_merkle_tree = if has_deferrals {
            let hasher = vm_poseidon2_hasher();
            let initial_memory = &self
                .app_prover
                .instance()
                .state()
                .as_ref()
                .expect("initial state should exist before proving")
                .memory
                .memory;
            Some(MerkleTree::from_memory(
                initial_memory,
                &memory_dimensions,
                &hasher,
            ))
        } else {
            None
        };

        let continuation_proof = self.app_prover.prove(vm_input)?;
        let (mut stark_proof, mut internal_metadata) =
            self.agg_prover.prove_vm(continuation_proof)?;

        let wrap_proof_type = if let Some(def_prover) = self.def_prover.as_ref() {
            let def_hook_proofs = def_prover.deferral_prover.prove(def_inputs)?;
            let def_proof = def_prover.agg_prover.prove_def(def_hook_proofs)?;
            stark_proof =
                self.agg_prover
                    .prove_mixed(stark_proof, def_proof, &mut internal_metadata)?;
            ProofsType::Combined
        } else {
            assert_eq!(def_inputs.len(), 0);
            ProofsType::Vm
        };

        // We add one additional internal_recursive layer to reduce the proof size.
        const ADDITIONAL_INTERNAL_RECURSIVE_LAYERS: usize = 1;
        for _ in 0..ADDITIONAL_INTERNAL_RECURSIVE_LAYERS {
            stark_proof =
                self.agg_prover
                    .wrap_proof(stark_proof, &mut internal_metadata, wrap_proof_type)?;
        }

        // Generate deferral merkle proofs if deferrals are enabled.
        if has_deferrals {
            let hasher = vm_poseidon2_hasher();
            let final_memory = &self
                .app_prover
                .instance()
                .state()
                .as_ref()
                .expect("final state should exist after proving")
                .memory
                .memory;
            let final_merkle_tree =
                MerkleTree::from_memory(final_memory, &memory_dimensions, &hasher);

            let def_pvs: &DeferralPvs<F> = stark_proof.inner.public_values[DEF_PVS_AIR_ID]
                .as_slice()
                .borrow();
            let depth = def_pvs.depth.as_canonical_u32() as usize;

            stark_proof.deferral_merkle_proofs = Some(compute_deferral_merkle_proofs(
                memory_dimensions,
                initial_merkle_tree.as_ref().unwrap(),
                &final_merkle_tree,
                depth,
            ));
        }

        Ok(stark_proof)
    }

    pub fn generate_baseline(&self) -> VerificationBaseline {
        VerificationBaseline {
            app_exe_commit: self.app_prover.app_exe_commit(),
            memory_dimensions: self.app_prover.memory_dimensions(),
            app_dag_commit: self.agg_prover.leaf_prover.get_dag_commit(false),
            leaf_dag_commit: self
                .agg_prover
                .internal_for_leaf_prover
                .get_dag_commit(false),
            internal_for_leaf_dag_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_dag_commit(false),
            internal_recursive_dag_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_dag_commit(true),
            expected_def_vk_commit: self.def_prover.as_ref().map(|dp| {
                let ir_dag_commit = dp
                    .deferral_prover
                    .internal_recursive_prover
                    .get_dag_commit(false);
                dp.deferral_prover.single_circuit_provers[0].vk_commit(ir_dag_commit)
            }),
        }
    }
}
