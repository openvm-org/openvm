use openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_continuations::verifier::{
    common::types::VmVerifierPvs, internal::types::InternalVmVerifierPvs,
};
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::{conversion::CompilerOptions, ir::DIGEST_SIZE};
use openvm_stark_sdk::config::FriParameters;
use serde::{Deserialize, Serialize};

mod global;
pub use global::*;

pub const DEFAULT_APP_LOG_BLOWUP: usize = 1;
pub const DEFAULT_LEAF_LOG_BLOWUP: usize = 1;
pub const DEFAULT_INTERNAL_LOG_BLOWUP: usize = 2;
pub const DEFAULT_ROOT_LOG_BLOWUP: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig<VC> {
    #[serde(default)]
    pub app_fri_params: AppFriParams,
    pub app_vm_config: VC,
    #[serde(default)]
    pub leaf_fri_params: LeafFriParams,
    /// Only for AggVM debugging. App VM users should not need this in regular flow.
    #[serde(default)]
    pub compiler_options: CompilerOptions,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AggConfig {
    /// STARK aggregation config
    pub agg_stark_config: AggStarkConfig,
    /// STARK-to-SNARK and SNARK-to-SNARK aggregation config
    pub halo2_config: Halo2Config,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AggStarkConfig {
    pub max_num_user_public_values: usize,
    pub leaf_fri_params: FriParameters,
    pub internal_fri_params: FriParameters,
    pub root_fri_params: FriParameters,
    /// Sets the profiling mode of all aggregation VMs
    pub profiling: bool,
    /// Only for AggVM debugging.
    pub compiler_options: CompilerOptions,
    /// Max constraint degree for FRI logup chunking
    pub root_max_constraint_degree: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Halo2Config {
    /// Log degree for the outer recursion verifier circuit.
    pub verifier_k: usize,
    /// If not specified, keygen will tune wrapper_k automatically.
    pub wrapper_k: Option<usize>,
    /// Sets the profiling mode of halo2 VM
    pub profiling: bool,
}

impl<VC> AppConfig<VC> {
    pub fn new(app_fri_params: FriParameters, app_vm_config: VC) -> Self {
        Self {
            app_fri_params: AppFriParams::from(app_fri_params),
            app_vm_config,
            leaf_fri_params: Default::default(),
            compiler_options: Default::default(),
        }
    }

    pub fn new_with_leaf_fri_params(
        app_fri_params: FriParameters,
        app_vm_config: VC,
        leaf_fri_params: FriParameters,
    ) -> Self {
        Self {
            app_fri_params: AppFriParams::from(app_fri_params),
            app_vm_config,
            leaf_fri_params: LeafFriParams::from(leaf_fri_params),
            compiler_options: Default::default(),
        }
    }
}

impl Default for AggStarkConfig {
    fn default() -> Self {
        Self {
            max_num_user_public_values: DEFAULT_MAX_NUM_PUBLIC_VALUES,
            leaf_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_LEAF_LOG_BLOWUP,
            ),
            internal_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_INTERNAL_LOG_BLOWUP,
            ),
            root_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_ROOT_LOG_BLOWUP,
            ),
            profiling: false,
            compiler_options: Default::default(),
            root_max_constraint_degree: (1 << DEFAULT_ROOT_LOG_BLOWUP) + 1,
        }
    }
}

impl Default for AggConfig {
    fn default() -> Self {
        Self {
            agg_stark_config: AggStarkConfig::default(),
            halo2_config: Halo2Config {
                verifier_k: 24,
                wrapper_k: None,
                profiling: false,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppFriParams {
    pub fri_params: FriParameters,
}

impl Default for AppFriParams {
    fn default() -> Self {
        Self {
            fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_APP_LOG_BLOWUP,
            ),
        }
    }
}

impl From<FriParameters> for AppFriParams {
    fn from(fri_params: FriParameters) -> Self {
        Self { fri_params }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeafFriParams {
    pub fri_params: FriParameters,
}

impl Default for LeafFriParams {
    fn default() -> Self {
        Self {
            fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_LEAF_LOG_BLOWUP,
            ),
        }
    }
}

impl From<FriParameters> for LeafFriParams {
    fn from(fri_params: FriParameters) -> Self {
        Self { fri_params }
    }
}

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
            // app_commit + leaf_verifier_commit + public_values
            DIGEST_SIZE * 2 + self.max_num_user_public_values,
            SBOX_SIZE.min(self.root_fri_params.max_constraint_degree()),
        );
        config.system.profiling = self.profiling;
        config
    }
}
