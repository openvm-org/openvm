use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, Streams, VirtualMachine, VmComplexTraceHeights,
        VmConfig,
    },
    system::{memory::tree::public_values::UserPublicValuesProof, program::trace::VmCommittedExe},
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::types::Proof,
    Chip,
};
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::prover::vm::{
    types::VmProvingKey, AsyncContinuationVmProver, AsyncSingleSegmentVmProver,
    ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
};

pub struct VmLocalProver<SC: StarkGenericConfig, VC, E: StarkFriEngine<SC>> {
    pub pk: Arc<VmProvingKey<SC, VC>>,
    pub committed_exe: Arc<VmCommittedExe<SC>>,
    overridden_heights: Option<VmComplexTraceHeights>,
    _marker: PhantomData<E>,
}

impl<SC: StarkGenericConfig, VC, E: StarkFriEngine<SC>> VmLocalProver<SC, VC, E> {
    pub fn new(pk: Arc<VmProvingKey<SC, VC>>, committed_exe: Arc<VmCommittedExe<SC>>) -> Self {
        Self {
            pk,
            committed_exe,
            overridden_heights: None,
            _marker: PhantomData,
        }
    }

    pub fn new_with_overridden_trace_heights(
        pk: Arc<VmProvingKey<SC, VC>>,
        committed_exe: Arc<VmCommittedExe<SC>>,
        overridden_heights: Option<VmComplexTraceHeights>,
    ) -> Self {
        Self {
            pk,
            committed_exe,
            overridden_heights,
            _marker: PhantomData,
        }
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.overridden_heights = Some(overridden_heights);
    }

    pub fn vm_config(&self) -> &VC {
        &self.pk.vm_config
    }
}

impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>> ContinuationVmProver<SC>
    for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> ContinuationVmProof<SC> {
        assert!(self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        let vm = VirtualMachine::new_with_overridden_trace_heights(
            e,
            self.pk.vm_config.clone(),
            self.overridden_heights.clone(),
        );
        let results = vm
            .execute_and_generate_with_cached_program(self.committed_exe.clone(), input)
            .unwrap();
        let user_public_values = UserPublicValuesProof::compute(
            self.pk.vm_config.system().memory_config.memory_dimensions(),
            self.pk.vm_config.system().num_public_values,
            &vm_poseidon2_hasher(),
            results.final_memory.as_ref().unwrap(),
        );
        let per_segment = vm.prove(&self.pk.vm_pk, results);
        ContinuationVmProof {
            per_segment,
            user_public_values,
        }
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncContinuationVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    async fn prove(
        &self,
        input: impl Into<Streams<Val<SC>>> + Send + Sync,
    ) -> ContinuationVmProof<SC> {
        ContinuationVmProver::prove(self, input)
    }
}

impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>> SingleSegmentVmProver<SC>
    for VmLocalProver<SC, VC, E>
where
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<Streams<Val<SC>>>) -> Proof<SC> {
        assert!(!self.pk.vm_config.system().continuation_enabled);
        let e = E::new(self.pk.fri_params);
        let vm = VirtualMachine::new(e, self.pk.vm_config.clone());
        let mut results = vm
            .execute_and_generate_with_cached_program(self.committed_exe.clone(), input)
            .unwrap();
        let segment = results.per_segment.pop().unwrap();
        vm.prove_single(&self.pk.vm_pk, segment)
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, VC: VmConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncSingleSegmentVmProver<SC> for VmLocalProver<SC, VC, E>
where
    VmLocalProver<SC, VC, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    async fn prove(&self, input: impl Into<Streams<Val<SC>>> + Send + Sync) -> Proof<SC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
