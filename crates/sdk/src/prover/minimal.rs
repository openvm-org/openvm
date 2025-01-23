use std::sync::Arc;

use openvm_circuit::arch::{SingleSegmentVmExecutor, Streams, VmConfig};
use openvm_native_circuit::NativeConfig;
use openvm_native_recursion::hints::Hintable;
use openvm_stark_backend::{prover::types::Proof, Chip};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2_root::BabyBearPoseidon2RootEngine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
};
use tracing::info_span;

use super::{
    app::AppProver,
    vm::{ContinuationVmProof, SingleSegmentVmProver},
    LeafProver, RootVerifierLocalProver, F, SC,
};
use crate::{
    keygen::{AppProvingKey, MinimalStarkProvingKey},
    verifier::{root::types::RootVmVerifierInput, utils::heights_le},
    NonRootCommittedExe, RootSC, StdIn,
};

pub struct MinimalProver<VC> {
    app_prover: AppProver<VC>,
    minimal_stark_prover: MinimalStarkProver,
}

impl<VC> MinimalProver<VC>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn new(
        app_exe: Arc<NonRootCommittedExe>,
        minimal_stark_pk: MinimalStarkProvingKey<VC>,
    ) -> Self {
        let app_vm_pk = minimal_stark_pk.app_vm_pk.clone();
        assert_eq!(
            app_vm_pk.fri_params, minimal_stark_pk.root_verifier_pk.vm_pk.fri_params,
            "App VM is incompatible with Minimal VM because of leaf FRI parameters"
        );
        assert_eq!(
            app_vm_pk.vm_config.system().num_public_values,
            minimal_stark_pk.num_public_values(),
            "App VM is incompatible with Minimal VM because of the number of public values"
        );

        Self {
            app_prover: AppProver::new(app_vm_pk.clone(), app_exe),
            minimal_stark_prover: MinimalStarkProver::new(
                minimal_stark_pk,
                // app_pk.leaf_committed_exe.clone(),
            ),
        }
    }

    pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
        self.app_prover.set_program_name(program_name);
        self
    }

    pub fn generate_proof_for_outer_recursion(&self, input: StdIn) -> Proof<RootSC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_proof = self.app_prover.generate_app_proof(input);
        self.minimal_stark_prover.generate_agg_proof(app_proof)
    }

    pub fn vm_config(&self) -> &NativeConfig {
        self.minimal_stark_prover.vm_config()
    }
}

pub struct MinimalStarkProver {
    pub root_prover: RootVerifierLocalProver,
}

impl MinimalStarkProver {
    pub fn new<VC: VmConfig<F>>(minimal_verifier_pk: MinimalStarkProvingKey<VC>) -> Self
    where
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        println!(
            "Creating MinimalStarkProver vmconfig: {:?}",
            minimal_verifier_pk.root_verifier_pk.vm_pk.vm_config
        );
        let root_prover = RootVerifierLocalProver::new(minimal_verifier_pk.root_verifier_pk);
        Self { root_prover }
    }

    pub fn generate_agg_proof(&self, app_proofs: ContinuationVmProof<SC>) -> Proof<RootSC> {
        let root_input = RootVmVerifierInput {
            proofs: app_proofs.per_segment.clone(),
            public_values: app_proofs.user_public_values.public_values.clone(),
        };

        // Validate proof heights before root proof generation
        let actual_air_heights = self.root_prover.execute_for_air_heights(root_input.clone());
        assert!(
            heights_le(
                &actual_air_heights,
                &self.root_prover.root_verifier_pk.air_heights
            ),
            "Proof heights exceed root verifier capacity"
        );

        self.generate_root_proof_impl(root_input)
    }

    fn generate_root_proof_impl(&self, root_input: RootVmVerifierInput<SC>) -> Proof<RootSC> {
        info_span!("minimal_stark_layer", group = "root", idx = 0).in_scope(|| {
            let input = root_input.write();
            #[cfg(feature = "bench-metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(self.root_prover.fri_params().log_blowup as u64);
            println!("Generating root proof: {:?}", input);
            SingleSegmentVmProver::prove(&self.root_prover, input)
        })
    }

    pub fn vm_config(&self) -> &NativeConfig {
        &self.root_prover.root_verifier_pk.vm_pk.vm_config
    }

    pub(crate) fn fri_params(&self) -> &FriParameters {
        &self.root_prover.root_verifier_pk.vm_pk.fri_params
    }
}

impl SingleSegmentVmProver<RootSC> for MinimalStarkProver {
    fn prove(&self, input: impl Into<Streams<F>>) -> Proof<RootSC> {
        let input = input.into();
        let vm = SingleSegmentVmExecutor::new(self.vm_config().clone());
        let mut proof_input = vm
            .execute_and_generate(
                self.root_prover.root_verifier_pk.root_committed_exe.clone(),
                input,
            )
            .unwrap();
        assert_eq!(
            proof_input.per_air.len(),
            self.root_prover.root_verifier_pk.air_heights.len(),
            "All AIRs of root verifier should present"
        );
        proof_input.per_air.iter().for_each(|(air_id, input)| {
            assert_eq!(
                input.main_trace_height(),
                self.root_prover.root_verifier_pk.air_heights[*air_id],
                "Trace height doesn't match"
            );
        });
        // Reorder the AIRs by heights.
        let air_id_perm = self.root_prover.root_verifier_pk.air_id_permutation();
        air_id_perm.permute(&mut proof_input.per_air);
        for i in 0..proof_input.per_air.len() {
            // Overwrite the AIR ID.
            proof_input.per_air[i].0 = i;
        }
        let e = BabyBearPoseidon2RootEngine::new(*self.fri_params());
        e.prove(&self.root_prover.root_verifier_pk.vm_pk.vm_pk, proof_input)
    }
}

pub struct MinimalRootVerifierProver {}
