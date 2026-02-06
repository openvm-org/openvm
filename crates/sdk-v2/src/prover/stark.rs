use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    Executor, MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig,
    instructions::exe::VmExe,
};
use openvm_stark_backend::{
    config::{Com, Val},
    p3_field::PrimeField32,
};
use stark_backend_v2::{StarkWhirEngine, poseidon2::CHUNK};
use verify_stark::{NonRootStarkProof, vk::VerificationBaseline};

use crate::{
    SC, StdIn,
    prover::{AggProver, AppProver, CompressionProver, vm::types::VmProvingKey},
};

pub struct StarkProver<E, VB>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
    pub compression_prover: Option<Arc<CompressionProver>>,
}

impl<E, VB> StarkProver<E, VB>
where
    E: StarkWhirEngine<SC = SC>,
    VB: VmBuilder<E>,
    Val<SC>: PrimeField32,
    Com<SC>: AsRef<[Val<SC>; CHUNK]> + From<[Val<SC>; CHUNK]> + Into<[Val<SC>; CHUNK]>,
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
        if let Some(compression_prover) = self.compression_prover.as_ref() {
            // We add two additional internal_recursive layers before the compression layer to
            // minimize the input size. The first internal_recursive layer will have a single
            // child proof, and thus may have 2-3x fewer trace cells than the previous layer.
            // The second will also only have a single child, and it may require 2-3x fewer
            // hashes (and thus Poseidon2 trace rows) than the first.
            const ADDITIONAL_INTERNAL_RECURSIVE_LAYERS: usize = 2;
            for _ in 0..ADDITIONAL_INTERNAL_RECURSIVE_LAYERS {
                stark_proof = self
                    .agg_prover
                    .wrap_proof(stark_proof, &mut internal_metadata)?;
            }
            stark_proof = compression_prover.prove(stark_proof)?;
        }
        Ok(stark_proof)
    }

    pub fn generate_baseline(&self) -> VerificationBaseline {
        VerificationBaseline {
            app_exe_commit: self.app_prover.app_exe_commit(),
            memory_dimensions: self.app_prover.memory_dimensions(),
            app_vk_commit: self.agg_prover.leaf_prover.get_cached_commit(false),
            leaf_vk_commit: self
                .agg_prover
                .internal_for_leaf_prover
                .get_cached_commit(false),
            internal_for_leaf_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_cached_commit(false),
            internal_recursive_vk_commit: self
                .agg_prover
                .internal_recursive_prover
                .get_cached_commit(true),
            compression_commit: self
                .compression_prover
                .as_ref()
                .map(|prover| prover.0.get_dag_commit()),
        }
    }
}
