use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, GenerationError, InsExecutorE1,
        SingleSegmentVmExecutor, Streams, VirtualMachine, VmComplexTraceHeights, VmConfig,
    },
    system::{
        memory::merkle::public_values::UserPublicValuesProof, program::trace::VmCommittedExe,
    },
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    proof::Proof,
    Chip,
};
use openvm_stark_sdk::{config::FriParameters, engine::StarkFriEngine};
use tracing::info_span;

use crate::prover::vm::{
    types::VmProvingKey, AsyncContinuationVmProver, AsyncSingleSegmentVmProver,
    ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
};

impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>> SingleSegmentVmProver<SC>
    for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + InsExecutorE1<Val<SC>>,
    VC::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> Proof<SC> {
        assert!(!self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        // note: use SingleSegmentVmExecutor so there's not a "segment" label in metrics
        let executor = {
            let mut executor = SingleSegmentVmExecutor::new(self.pk.vm_config.clone());
            executor.set_trace_height_constraints(self.pk.vm_pk.trace_height_constraints.clone());
            executor
        };

        let vm_vk = self.pk.vm_pk.get_vk();
        let input = input.into();
        let max_trace_heights = if let Some(overridden_heights) = &self.single_segment_heights {
            overridden_heights
        } else {
            &executor
                .execute_metered(
                    self.committed_exe.exe.clone(),
                    input.clone(),
                    &vm_vk.total_widths(),
                    &vm_vk.num_interactions(),
                )
                .expect("execute_metered failed")
        };
        let proof_input = executor
            .execute_and_generate(self.committed_exe.clone(), input, max_trace_heights)
            .unwrap();

        let vm = VirtualMachine::new(e, executor.config);
        vm.prove_single(&self.pk.vm_pk, proof_input)
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncSingleSegmentVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC> + InsExecutorE1<Val<SC>>,
    VC::Periphery: Chip<SC>,
{
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
