use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor, VmBuilder,
    VmExecutionConfig,
};
use openvm_continuations::circuit::inner::ProofsType;
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, Val};
use openvm_verify_stark_host::{vk::VerificationBaseline, NonRootStarkProof};

use crate::{
    prover::{vm::types::VmProvingKey, AggProver, AppProver, DeferralProver},
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
        let continuation_proof = self.app_prover.prove(vm_input)?;
        let (mut stark_proof, mut internal_metadata) =
            self.agg_prover.prove_vm(continuation_proof)?;

        if let Some(def_prover) = self.def_prover.as_ref() {
            let def_hook_proofs = def_prover.deferral_prover.prove(def_inputs)?;
            let def_proof = def_prover.agg_prover.prove_def(def_hook_proofs)?;
            stark_proof =
                self.agg_prover
                    .prove_mixed(stark_proof, def_proof, &mut internal_metadata)?;
        } else {
            assert_eq!(def_inputs.len(), 0);
        }

        // We add one additional internal_recursive layer to reduce the proof size.
        const ADDITIONAL_INTERNAL_RECURSIVE_LAYERS: usize = 1;
        for _ in 0..ADDITIONAL_INTERNAL_RECURSIVE_LAYERS {
            stark_proof =
                self.agg_prover
                    .wrap_proof(stark_proof, &mut internal_metadata, ProofsType::Vm)?;
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
        }
    }
}
