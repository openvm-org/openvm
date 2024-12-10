use std::sync::Arc;

use ax_stark_backend::{prover::types::Proof, Chip};
use axvm_circuit::arch::VmConfig;

use crate::{
    keygen::{AggProvingKey, AppProvingKey},
    prover::{agg::AggStarkProver, app::AppProver},
    NonRootCommittedExe, RootSC, StdIn, F, SC,
};

pub struct StarkProver<VC> {
    app_prover: AppProver<VC>,
    agg_prover: AggStarkProver,
}
impl<VC> StarkProver<VC> {
    pub fn new(
        app_pk: AppProvingKey<VC>,
        app_committed_exe: Arc<NonRootCommittedExe>,
        agg_pk: AggProvingKey,
    ) -> Self
    where
        VC: VmConfig<F>,
    {
        let AppProvingKey {
            leaf_committed_exe,
            leaf_fri_params,
            app_vm_pk,
        } = app_pk;
        assert_eq!(
            leaf_fri_params, agg_pk.leaf_vm_pk.fri_params,
            "App VM is incompatible with Agg VM because of leaf FRI parameters"
        );
        assert_eq!(
            app_vm_pk.vm_config.system().num_public_values,
            agg_pk.num_public_values(),
            "App VM is incompatible with Agg VM  because of the number of public values"
        );

        Self {
            app_prover: AppProver::new(app_vm_pk, app_committed_exe),
            agg_prover: AggStarkProver::new(agg_pk, leaf_committed_exe),
        }
    }
    pub fn generate_proof_for_outer_recursion(&self, input: StdIn) -> Proof<RootSC>
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let app_proof = self.app_prover.generate_app_proof(input);
        self.agg_prover.generate_agg_proof(app_proof)
    }
}
