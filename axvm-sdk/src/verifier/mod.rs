use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::ir::DIGEST_SIZE;
use internal::types::InternalVmVerifierPvs;

use crate::{config::AxVmSdkConfig, verifier::common::types::VmVerifierPvs};

pub mod common;
pub mod internal;
pub mod leaf;
pub mod root;
pub(crate) mod utils;

const SBOX_SIZE: usize = 7;

impl AxVmSdkConfig {
    pub fn leaf_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            VmVerifierPvs::<u8>::width(),
            SBOX_SIZE.min(self.leaf_fri_params.max_constraint_degree()),
        )
    }
    pub fn internal_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            InternalVmVerifierPvs::<u8>::width(),
            SBOX_SIZE.min(self.internal_fri_params.max_constraint_degree()),
        )
    }
    pub fn root_verifier_vm_config(&self) -> VmConfig {
        VmConfig::aggregation(
            // app_commit + leaf_verifier_commit + public_values
            DIGEST_SIZE * 2 + self.max_num_user_public_values,
            SBOX_SIZE.min(self.root_fri_params.max_constraint_degree()),
        )
    }
}
