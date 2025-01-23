use internal::types::InternalVmVerifierPvs;
use openvm_circuit::arch::VmConfig;
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::Chip;
use openvm_stark_sdk::config::FriParameters;

use crate::{
    config::{AggStarkConfig, MinimalStarkConfig},
    verifier::common::types::VmVerifierPvs,
    F, SC,
};

pub mod common;
pub mod internal;
pub mod leaf;
pub mod minimal;
pub mod root;
pub mod utils;

const SBOX_SIZE: usize = 7;

impl AggStarkConfig {
    pub fn leaf_vm_config(&self) -> NativeConfig {
        let mut config = NativeConfig::aggregation(
            VmVerifierPvs::<u8>::width(),
            SBOX_SIZE.min(self.leaf_fri_params.max_constraint_degree()),
        );
        config.system.profiling = self.profiling;
        config
    }
    pub fn internal_vm_config(&self) -> NativeConfig {
        let mut config = NativeConfig::aggregation(
            InternalVmVerifierPvs::<u8>::width(),
            SBOX_SIZE.min(self.internal_fri_params.max_constraint_degree()),
        );
        config.system.profiling = self.profiling;
        config
    }
    pub fn root_verifier_vm_config(&self) -> NativeConfig {
        let mut config = NativeConfig::aggregation(
            DIGEST_SIZE * 2 + self.max_num_user_public_values,
            SBOX_SIZE.min(self.root_fri_params.max_constraint_degree()),
        );
        config.system.profiling = self.profiling;
        config
    }
}

impl<VC> MinimalStarkConfig<VC>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    pub fn minimal_root_verifier_vm_config(&self) -> NativeConfig {
        let mut config = NativeConfig::aggregation(
            DIGEST_SIZE * 2 + self.max_num_user_public_values,
            // VmVerifierPvs::<u8>::width(),
            SBOX_SIZE.min(self.root_fri_params.max_constraint_degree()),
        );
        config.system.profiling = self.profiling;
        config
    }
}
