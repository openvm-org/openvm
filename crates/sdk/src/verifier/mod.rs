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
pub(crate) mod utils;

const SBOX_SIZE: usize = 7;

impl AggStarkConfig {
    pub fn leaf_vm_config(&self) -> NativeConfig {
        leaf_vm_config(&self.leaf_fri_params, self.profiling)
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
        root_verifier_vm_config(
            &self.root_fri_params,
            self.max_num_user_public_values,
            self.profiling,
        )
    }
}

impl<VC> MinimalStarkConfig<VC>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    // pub fn leaf_vm_config(&self) -> NativeConfig {
    //     leaf_vm_config(&self.leaf_fri_params, self.profiling)
    // }

    pub fn root_verifier_vm_config(&self) -> NativeConfig {
        root_verifier_vm_config(
            &self.root_fri_params,
            self.max_num_user_public_values,
            self.profiling,
        )
    }
}

fn leaf_vm_config(fri_params: &FriParameters, profiling: bool) -> NativeConfig {
    let mut config = NativeConfig::aggregation(
        VmVerifierPvs::<u8>::width(),
        SBOX_SIZE.min(fri_params.max_constraint_degree()),
    );
    config.system.profiling = profiling;
    config
}

fn root_verifier_vm_config(
    fri_params: &FriParameters,
    max_num_user_public_values: usize,
    profiling: bool,
) -> NativeConfig {
    let mut config = NativeConfig::aggregation(
        DIGEST_SIZE * 2 + max_num_user_public_values,
        SBOX_SIZE.min(fri_params.max_constraint_degree()),
    );
    config.system.profiling = profiling;
    config
}
