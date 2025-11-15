use std::sync::Arc;

use continuations_v2::aggregation::{AggregationProver, NonRootStarkProof};
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
use verify_stark::VerificationBaseline;

use crate::{
    SC, StdIn,
    prover::{AggProver, AppProver, vm::types::VmProvingKey},
};

pub struct StarkProver<E, VB>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
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
    ) -> Result<Self> {
        Ok(Self {
            app_prover: AppProver::new(vm_builder, app_vm_pk, app_exe)?,
            agg_prover,
        })
    }

    pub fn prove(&mut self, input: StdIn<Val<SC>>) -> Result<NonRootStarkProof>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let continuation_proof = self.app_prover.prove(input)?;
        self.agg_prover.prove(continuation_proof)
    }

    pub fn generate_baseline(&self) -> VerificationBaseline {
        VerificationBaseline {
            app_exe_commit: self.app_prover.app_exe_commit(),
            memory_dimensions: self.app_prover.memory_dimensions(),
            leaf_commit: self.agg_prover.leaf_prover.get_commit(),
            internal_for_leaf_commit: self.agg_prover.internal_for_leaf_prover.get_commit(),
            internal_recursive_commit: self.agg_prover.internal_recursive_prover.get_commit(),
        }
    }
}
