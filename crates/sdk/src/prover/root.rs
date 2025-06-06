use async_trait::async_trait;
use itertools::Itertools;
use openvm_circuit::arch::{
    execution_mode::metered::get_widths_and_interactions_from_vkey, SingleSegmentVmExecutor,
    Streams,
};
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
use openvm_native_circuit::NativeConfig;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2_root::BabyBearPoseidon2RootEngine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::proof::Proof,
};

use crate::{
    keygen::RootVerifierProvingKey,
    prover::vm::{AsyncSingleSegmentVmProver, SingleSegmentVmProver},
    RootSC, F, SC,
};

/// Local prover for a root verifier.
pub struct RootVerifierLocalProver {
    pub root_verifier_pk: RootVerifierProvingKey,
    executor_for_heights: SingleSegmentVmExecutor<F, NativeConfig>,
}

impl RootVerifierLocalProver {
    pub fn new(root_verifier_pk: RootVerifierProvingKey) -> Self {
        let executor_for_heights =
            SingleSegmentVmExecutor::<F, _>::new(root_verifier_pk.vm_pk.vm_config.clone());
        Self {
            root_verifier_pk,
            executor_for_heights,
        }
    }
    pub fn execute_for_air_heights(&self, input: RootVmVerifierInput<SC>) -> Vec<usize> {
        let (widths, interactions) =
            get_widths_and_interactions_from_vkey(self.root_verifier_pk.vm_pk.vm_pk.get_vk());
        let max_trace_heights = self
            .executor_for_heights
            .execute_metered(
                self.root_verifier_pk.root_committed_exe.exe.clone(),
                input.write(),
                widths,
                interactions,
            )
            .unwrap();
        let result = self
            .executor_for_heights
            .execute_with_max_heights_and_compute_heights(
                self.root_verifier_pk.root_committed_exe.exe.clone(),
                input.write(),
                &max_trace_heights,
            )
            .unwrap();
        result.air_heights
    }
    pub fn vm_config(&self) -> &NativeConfig {
        &self.root_verifier_pk.vm_pk.vm_config
    }
    #[allow(dead_code)]
    pub(crate) fn fri_params(&self) -> &FriParameters {
        &self.root_verifier_pk.vm_pk.fri_params
    }
}

impl SingleSegmentVmProver<RootSC> for RootVerifierLocalProver {
    fn prove(&self, input: impl Into<Streams<F>>) -> Proof<RootSC> {
        let input = input.into();
        let mut vm = SingleSegmentVmExecutor::new(self.vm_config().clone());
        vm.set_override_trace_heights(self.root_verifier_pk.vm_heights.clone());
        let trace_heights = self
            .root_verifier_pk
            .air_heights
            .iter()
            .map(|&height| height as u32)
            .collect_vec();
        let mut proof_input = vm
            .execute_with_max_heights_and_generate(
                self.root_verifier_pk.root_committed_exe.clone(),
                input,
                &trace_heights,
            )
            .unwrap();
        assert_eq!(
            proof_input.per_air.len(),
            self.root_verifier_pk.air_heights.len(),
            "All AIRs of root verifier should present"
        );
        proof_input.per_air.iter().for_each(|(air_id, input)| {
            assert_eq!(
                input.main_trace_height(),
                self.root_verifier_pk.air_heights[*air_id],
                "Trace height doesn't match"
            );
        });
        // Reorder the AIRs by heights.
        let air_id_perm = self.root_verifier_pk.air_id_permutation();
        air_id_perm.permute(&mut proof_input.per_air);
        for i in 0..proof_input.per_air.len() {
            // Overwrite the AIR ID.
            proof_input.per_air[i].0 = i;
        }
        let e = BabyBearPoseidon2RootEngine::new(*self.fri_params());
        e.prove(&self.root_verifier_pk.vm_pk.vm_pk, proof_input)
    }
}

#[async_trait]
impl AsyncSingleSegmentVmProver<RootSC> for RootVerifierLocalProver {
    async fn prove(&self, input: impl Into<Streams<F>> + Send + Sync) -> Proof<RootSC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
