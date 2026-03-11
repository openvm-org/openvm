use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor, VmBuilder,
    VmExecutionConfig,
};
use openvm_stark_backend::{p3_field::PrimeField32, StarkEngine, Val};
use verify_stark::{vk::VerificationBaseline, NonRootStarkProof};

use crate::{
    prover::{vm::types::VmProvingKey, AggProver, AppProver, CompressionProver},
    StdIn, SC,
};

pub struct StarkProver<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
    pub compression_prover: Option<Arc<CompressionProver>>,
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
        compression_prover: Option<Arc<CompressionProver>>,
    ) -> Result<Self> {
        Ok(Self {
            app_prover: AppProver::new(vm_builder, app_vm_pk, app_exe)?,
            agg_prover,
            compression_prover,
        })
    }

    pub fn prove(&mut self, input: StdIn<Val<SC>>) -> Result<NonRootStarkProof>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let continuation_proof = self.app_prover.prove(input)?;
        let (mut stark_proof, mut internal_metadata) = self.agg_prover.prove(continuation_proof)?;
        // We add one additional internal_recursive layer to reduce the proof size.
        const ADDITIONAL_INTERNAL_RECURSIVE_LAYERS: usize = 1;
        for _ in 0..ADDITIONAL_INTERNAL_RECURSIVE_LAYERS {
            stark_proof = self
                .agg_prover
                .wrap_proof(stark_proof, &mut internal_metadata)?;
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
            compression_commit: self
                .compression_prover
                .as_ref()
                .map(|prover| prover.0.get_dag_commit()),
        }
    }
}
