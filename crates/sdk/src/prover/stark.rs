use std::sync::Arc;

use openvm_circuit::arch::VmConfig;
use openvm_continuations::verifier::{
    internal::types::VmStarkProof, root::types::RootVmVerifierInput,
};
use openvm_stark_backend::{proof::Proof, Chip};
use openvm_stark_sdk::engine::StarkFriEngine;

use crate::{
    config::AggregationTreeConfig,
    keygen::{AggStarkProvingKey, AppProvingKey},
    prover::{agg::AggStarkProver, app::AppProver},
    NonRootCommittedExe, RootSC, StdIn, F, SC,
};

pub struct StarkProver<VC, E: StarkFriEngine<SC>> {
    pub app_prover: AppProver<VC, E>,
    pub agg_prover: AggStarkProver<E>,
}
impl<VC, E: StarkFriEngine<SC>> StarkProver<VC, E> {
    pub fn new(
        app_pk: Arc<AppProvingKey<VC>>,
        app_committed_exe: Arc<NonRootCommittedExe>,
        agg_stark_pk: AggStarkProvingKey,
        agg_tree_config: AggregationTreeConfig,
    ) -> Self
    where
        VC: VmConfig<F>,
    {
        assert_eq!(
            app_pk.leaf_fri_params, agg_stark_pk.leaf_vm_pk.fri_params,
            "App VM is incompatible with Agg VM because of leaf FRI parameters"
        );
        assert_eq!(
            app_pk.app_vm_pk.vm_config.system().num_public_values,
            agg_stark_pk.num_user_public_values(),
            "App VM is incompatible with Agg VM  because of the number of public values"
        );

        Self {
            app_prover: AppProver::new(app_pk.app_vm_pk.clone(), app_committed_exe),
            agg_prover: AggStarkProver::new(
                agg_stark_pk,
                app_pk.leaf_committed_exe.clone(),
                agg_tree_config,
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
        self.agg_prover.generate_root_proof(app_proof)
    }

    pub fn generate_root_verifier_input(&self, input: StdIn) -> RootVmVerifierInput<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_proof = self.app_prover.generate_app_proof(input);
        self.agg_prover.generate_root_verifier_input(app_proof)
    }

    pub fn generate_e2e_stark_proof(&self, input: StdIn) -> VmStarkProof<SC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_proof = self.app_prover.generate_app_proof(input);
        let leaf_proofs = self.agg_prover.generate_leaf_proofs(&app_proof);
        self.agg_prover
            .aggregate_leaf_proofs(leaf_proofs, app_proof.user_public_values.public_values)
    }
}
