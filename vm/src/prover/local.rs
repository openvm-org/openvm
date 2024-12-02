use std::{collections::VecDeque, marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    prover::types::Proof,
    Chip,
};
use ax_stark_sdk::engine::StarkFriEngine;
use p3_field::PrimeField32;

use crate::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, new_vm::VirtualMachine, VmComplexTraceHeights,
        VmGenericConfig,
    },
    prover::{
        types::VmProvingKey, AsyncContinuationVmProver, AsyncSingleSegmentVmProver,
        ContinuationVmProof, ContinuationVmProver, SingleSegmentVmProver,
    },
    system::{
        memory::tree::public_values::UserPublicValuesProof, program::trace::AxVmCommittedExe,
    },
};

pub struct VmLocalProver<SC: StarkGenericConfig, VmConfig, E: StarkFriEngine<SC>> {
    pub pk: VmProvingKey<SC, VmConfig>,
    pub committed_exe: Arc<AxVmCommittedExe<SC>>,
    overridden_heights: Option<VmComplexTraceHeights>,
    _marker: PhantomData<E>,
}

impl<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>, E: StarkFriEngine<SC>>
    VmLocalProver<SC, VmConfig, E>
where
    Val<SC>: PrimeField32,
{
    pub fn new(pk: VmProvingKey<SC, VmConfig>, committed_exe: Arc<AxVmCommittedExe<SC>>) -> Self {
        Self {
            pk,
            committed_exe,
            overridden_heights: None,
            _marker: PhantomData,
        }
    }

    pub fn new_with_overridden_trace_heights(
        pk: VmProvingKey<SC, VmConfig>,
        committed_exe: Arc<AxVmCommittedExe<SC>>,
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
}

impl<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>, E: StarkFriEngine<SC>>
    ContinuationVmProver<SC> for VmLocalProver<SC, VmConfig, E>
where
    Val<SC>: PrimeField32,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>>) -> ContinuationVmProof<SC> {
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
impl<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncContinuationVmProver<SC> for VmLocalProver<SC, VmConfig, E>
where
    VmLocalProver<SC, VmConfig, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    async fn prove(
        &self,
        input: impl Into<VecDeque<Vec<Val<SC>>>> + Send + Sync,
    ) -> ContinuationVmProof<SC> {
        ContinuationVmProver::prove(self, input)
    }
}

impl<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>, E: StarkFriEngine<SC>>
    SingleSegmentVmProver<SC> for VmLocalProver<SC, VmConfig, E>
where
    Val<SC>: PrimeField32,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>>) -> Proof<SC> {
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
impl<SC: StarkGenericConfig, VmConfig: VmGenericConfig<Val<SC>>, E: StarkFriEngine<SC>>
    AsyncSingleSegmentVmProver<SC> for VmLocalProver<SC, VmConfig, E>
where
    VmLocalProver<SC, VmConfig, E>: Send + Sync,
    Val<SC>: PrimeField32,
    VmConfig::Executor: Chip<SC>,
    VmConfig::Periphery: Chip<SC>,
{
    async fn prove(&self, input: impl Into<VecDeque<Vec<Val<SC>>>> + Send + Sync) -> Proof<SC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
