use std::env;

use openvm_sdk::config::{AppConfig, SdkVmConfig, DEFAULT_APP_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP};
use openvm_stark_sdk::config::FriParameters;

pub const DEFAULT_MANIFEST_DIR: &str = ".";

pub const DEFAULT_APP_CONFIG_PATH: &str = "./openvm.toml";
pub const DEFAULT_APP_EXE_PATH: &str = "./openvm/app.vmexe";
pub const DEFAULT_COMMITTED_APP_EXE_PATH: &str = "./openvm/committed_app_exe.bc";
pub const DEFAULT_APP_PK_PATH: &str = "./openvm/app.pk";
pub const DEFAULT_APP_VK_PATH: &str = "./openvm/app.vk";
pub const DEFAULT_APP_PROOF_PATH: &str = "./openvm/app.proof";
pub const DEFAULT_EVM_PROOF_PATH: &str = "./openvm/evm.proof";

pub fn default_agg_stark_pk_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/agg_stark.pk"
}

pub fn default_agg_halo2_pk_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/agg_halo2.pk"
}

pub fn default_asm_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/root.asm"
}

pub fn default_params_dir() -> String {
    env::var("HOME").unwrap() + "/.openvm/params/"
}

pub fn default_evm_halo2_verifier_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/halo2/"
}

pub fn default_app_config() -> AppConfig<SdkVmConfig> {
    AppConfig {
        app_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
            DEFAULT_APP_LOG_BLOWUP,
        )
        .into(),
        app_vm_config: SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .build(),
        leaf_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
            DEFAULT_LEAF_LOG_BLOWUP,
        )
        .into(),
        compiler_options: Default::default(),
    }
}
