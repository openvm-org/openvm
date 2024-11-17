use ax_stark_sdk::config::FriParameters;
use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::conversion::CompilerOptions;

#[derive(Clone, Debug)]
pub struct AxVmSdkConfig {
    pub max_num_user_public_values: usize,
    pub app_fri_params: FriParameters,
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub root_fri_params: FriParameters,
    pub app_vm_config: VmConfig,
    pub compiler_options: CompilerOptions,
}
