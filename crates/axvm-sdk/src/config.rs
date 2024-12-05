use ax_stark_sdk::config::FriParameters;
use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use serde::{Deserialize, Serialize};

use crate::F;

#[derive(Clone, Debug)]
pub struct AppConfig<VC: VmConfig<F>> {
    pub app_fri_params: FriParameters,
    pub app_vm_config: VC,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AggConfig {
    pub max_num_user_public_values: usize,
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub root_fri_params: FriParameters,
    pub compiler_options: CompilerOptions,
}
