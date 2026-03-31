use std::env;

use openvm_sdk::config::AppConfig;
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_sdk::config::{app_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT};

pub const DEFAULT_MANIFEST_DIR: &str = ".";

pub const DEFAULT_APP_PK_NAME: &str = "app.pk";
pub const DEFAULT_APP_VK_NAME: &str = "app.vk";

pub fn default_params_dir() -> String {
    env::var("HOME").unwrap() + "/.openvm/params/"
}

pub fn default_root_pk_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/root.pk"
}

pub fn default_evm_halo2_verifier_path() -> String {
    env::var("HOME").unwrap() + "/.openvm/halo2/"
}

pub fn default_app_config() -> AppConfig<SdkVmConfig> {
    AppConfig {
        app_vm_config: SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .build(),
        system_params: app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT),
    }
}
