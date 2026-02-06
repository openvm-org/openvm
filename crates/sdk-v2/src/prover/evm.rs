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
use stark_backend_v2::{StarkWhirEngine, poseidon2::CHUNK, proof::Proof};

use crate::{
    SC, StdIn,
    prover::{AggProver, AppProver, RootProver, vm::types::VmProvingKey},
};

pub struct EvmProver<E, VB>
where
    E: StarkWhirEngine,
    VB: VmBuilder<E>,
{
    pub app_prover: AppProver<E, VB>,
    pub agg_prover: Arc<AggProver>,
    pub root_prover: Arc<RootProver>,
}

impl<E, VB> EvmProver<E, VB>
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
        root_prover: Arc<RootProver>,
    ) -> Result<Self> {
        Ok(Self {
            app_prover: AppProver::new(vm_builder, app_vm_pk, app_exe)?,
            agg_prover,
            root_prover,
        })
    }

    // TODO[INT-5581]: should output an EvmProof
    pub fn prove(&mut self, input: StdIn<Val<SC>>) -> Result<Proof>
    where
        <VB::VmConfig as VmExecutionConfig<Val<SC>>>::Executor: Executor<Val<SC>>
            + MeteredExecutor<Val<SC>>
            + PreflightExecutor<Val<SC>, VB::RecordArena>,
    {
        let continuation_proof = self.app_prover.prove(input)?;
        let (stark_proof, _internal_metadata) = self.agg_prover.prove(continuation_proof)?;
        // TODO[INT-6027]: Wrap stark_proof until root_proof can be constant-sized
        let root_proof = self.root_prover.prove(stark_proof)?;
        Ok(root_proof)
    }
}
