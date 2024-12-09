use std::sync::Arc;

use ax_stark_sdk::ax_stark_backend::Chip;
use axvm_circuit::arch::VmConfig;
use axvm_native_recursion::halo2::EvmProof;

use crate::{io::StdIn, keygen::AppProvingKey, NonRootCommittedExe, F, SC};

pub mod agg;
pub mod app;
pub mod halo2;
mod root;
pub mod stark;

pub use root::*;

use crate::{
    keygen::FullAggProvingKey,
    prover::{halo2::Halo2Prover, stark::StarkProver},
};

pub struct ContinuationProver<VC> {
    stark_prover: StarkProver<VC>,
    halo2_prover: Halo2Prover,
}

impl<VC> ContinuationProver<VC> {
    pub fn new(
        app_pk: AppProvingKey<VC>,
        app_committed_exe: Arc<NonRootCommittedExe>,
        full_agg_pk: FullAggProvingKey,
    ) -> Self
    where
        VC: VmConfig<F>,
    {
        let FullAggProvingKey {
            agg_vm_pk,
            halo2_pk,
        } = full_agg_pk;
        let stark_prover = StarkProver::new(app_pk, app_committed_exe, agg_vm_pk);
        Self {
            stark_prover,
            halo2_prover: Halo2Prover::new(halo2_pk),
        }
    }
    pub fn generate_proof_for_evm(&self, input: StdIn) -> EvmProof
    where
        VC: VmConfig<F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let root_proof = self.stark_prover.generate_e2e_proof(input);
        self.halo2_prover.prove_for_evm(&root_proof)
    }
}
