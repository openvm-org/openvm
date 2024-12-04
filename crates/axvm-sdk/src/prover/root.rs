use std::collections::VecDeque;

use async_trait::async_trait;
use ax_stark_sdk::{
    ax_stark_backend::prover::types::Proof,
    config::baby_bear_poseidon2_outer::BabyBearPoseidon2OuterEngine,
    engine::{StarkEngine, StarkFriEngine},
};
use axvm_circuit::{
    arch::SingleSegmentVmExecutor,
    prover::{AsyncSingleSegmentVmProver, SingleSegmentVmProver},
};
use axvm_native_circuit::NativeConfig;
use axvm_native_recursion::hints::Hintable;

use crate::{
    keygen::RootVerifierProvingKey, verifier::root::types::RootVmVerifierInput, OuterSC, F, SC,
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
        let result = self
            .executor_for_heights
            .execute(
                self.root_verifier_pk.root_committed_exe.exe.clone(),
                input.write(),
            )
            .unwrap();
        result.air_heights
    }
}

impl SingleSegmentVmProver<OuterSC> for RootVerifierLocalProver {
    fn prove(&self, input: impl Into<VecDeque<Vec<F>>>) -> Proof<OuterSC> {
        let input = input.into();
        let vm = SingleSegmentVmExecutor::new(self.root_verifier_pk.vm_pk.vm_config.clone());
        let mut proof_input = vm
            .execute_and_generate(
                self.root_verifier_pk.root_committed_exe.clone(),
                input.into(),
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
        let e = BabyBearPoseidon2OuterEngine::new(self.root_verifier_pk.vm_pk.fri_params);
        e.prove(&self.root_verifier_pk.vm_pk.vm_pk, proof_input)
    }
}

#[async_trait]
impl AsyncSingleSegmentVmProver<OuterSC> for RootVerifierLocalProver {
    async fn prove(&self, input: impl Into<VecDeque<Vec<F>>> + Send + Sync) -> Proof<OuterSC> {
        SingleSegmentVmProver::prove(self, input)
    }
}
