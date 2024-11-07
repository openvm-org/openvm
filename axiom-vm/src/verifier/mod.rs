use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::ir::DIGEST_SIZE;

use crate::{config::AxiomVmConfig, verifier::internal::types::InternalVmVerifierPvs};

pub mod common;
pub mod internal;
pub mod leaf;
pub mod root;
pub(crate) mod utils;

impl AxiomVmConfig {
    pub fn non_root_verifier_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            // InternalVmVerifierPvs has more public values than the number of public values in the leaf verifier.
            InternalVmVerifierPvs::<u8>::width(),
            self.poseidon2_max_constraint_degree,
        )
    }
    pub fn root_verifier_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            // app_commit + leaf_verifier_commit + public_values
            DIGEST_SIZE * 2 + self.max_num_user_public_values,
            self.poseidon2_max_constraint_degree,
        )
    }
}
