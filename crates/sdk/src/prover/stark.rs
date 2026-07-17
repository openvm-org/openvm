use std::{borrow::Borrow, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, Executor,
        MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig,
    },
    system::memory::merkle::MerkleTree,
};
use openvm_stark_backend::{p3_field::PrimeField32, prover::ProverDevice, StarkEngine, Val};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};
use openvm_verify_stark_host::{
    pvs::{DeferralPvs, DEF_PVS_AIR_ID},
    vk::VerificationBaseline,
    VmStarkProof,
};

use crate::{
    prover::{
        deferral::compute_deferral_merkle_proofs, vm::types::VmProvingKey, AggProver, AppProver,
        InternalLayerMetadata,
    },
    DeferralInput, DeferralSetup, StdIn, SC,
};

pub struct StarkProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
    pub deferral_setup: DeferralSetup,
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
        deferral_setup: DeferralSetup,
    ) -> Result<Self> {
        Ok(Self {
            app_prover: AppProver::new(vm_builder, app_vm_pk, app_exe)?,
            agg_prover,
            deferral_setup,
        })
    }

    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.app_prover.set_program_name(program_name);
        self
    }

    pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
        self.set_program_name(program_name);
        self
    }

    pub fn prove(
        &mut self,
        vm_input: StdIn,
        def_inputs: &[DeferralInput],
    ) -> Result<(VmStarkProof, InternalLayerMetadata)>
    where
        <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let has_deferrals = self.deferral_setup.hook_commit().is_some();
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

        // Skip aggregation unless some circuit received a deferred call. Note that
        // deferrals are also skipped if def_inputs is an empty slice.
        if def_inputs.iter().any(|input| !input.is_empty()) {
            let def_agg_prover = self.deferral_setup.prover().ok_or_else(|| {
                eyre::eyre!("non-empty deferral inputs require a deferral aggregation prover")
            })?;
            let def_hook_proofs = def_agg_prover
                .multi_deferral_circuit_prover
                .prove(def_inputs)?;
            let (def_proof, def_internal_recursive_layer) =
                def_agg_prover.agg_prover.prove_def(def_hook_proofs)?;
            stark_proof = self.agg_prover.prove_mixed(
                stark_proof,
                def_proof,
                &mut internal_metadata,
                def_internal_recursive_layer,
            )?;
        }

        // We add one additional internal_recursive layer to reduce the proof size.
        const ADDITIONAL_INTERNAL_RECURSIVE_LAYERS: usize = 1;
        for _ in 0..ADDITIONAL_INTERNAL_RECURSIVE_LAYERS {
            stark_proof = self
                .agg_prover
                .wrap_proof(stark_proof, &mut internal_metadata)?;
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

        Ok((stark_proof, internal_metadata))
    }

    pub fn generate_baseline(&self) -> VerificationBaseline {
        VerificationBaseline {
            app_exe_commit: self.app_prover.app_exe_commit(),
            memory_dimensions: self.app_prover.memory_dimensions(),
            num_user_pvs: self.app_prover.num_user_pvs(),
            app_vk_commit: self.agg_prover.leaf_prover.get_vk_commit(false),
            leaf_vk_commit: self
                .agg_prover
                .internal_for_leaf_prover
                .get_vk_commit(false),
            internal_for_leaf_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_vk_commit(false),
            internal_recursive_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_vk_commit(true),
            expected_def_hook_commit: self.deferral_setup.hook_commit(),
        }
    }

    pub fn app_vm_commit(&self) -> Digest {
        self.agg_prover.vm_or_hook_commit()
    }
}
