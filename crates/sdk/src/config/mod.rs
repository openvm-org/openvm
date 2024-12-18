use openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_stark_sdk::config::FriParameters;
use serde::{Deserialize, Serialize};

mod global;
pub use global::*;

const DEFAULT_APP_BLOWUP: usize = 2;
const DEFAULT_LEAF_BLOWUP: usize = 2;
const DEFAULT_INTERNAL_BLOWUP: usize = 2;
const DEFAULT_ROOT_BLOWUP: usize = 3;

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
    /// Only for AggVM debugging.
    pub compiler_options: CompilerOptions,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Halo2Config {
    /// Log degree for the outer recursion verifier circuit.
    pub verifier_k: usize,
    /// If not specified, keygen will tune wrapper_k automatically.
    pub wrapper_k: Option<usize>,
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
                DEFAULT_LEAF_BLOWUP,
            ),
            internal_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_INTERNAL_BLOWUP,
            ),
            root_fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_ROOT_BLOWUP,
            ),
            compiler_options: Default::default(),
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
                DEFAULT_APP_BLOWUP,
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
                DEFAULT_LEAF_BLOWUP,
            ),
        }
    }
}

impl From<FriParameters> for LeafFriParams {
    fn from(fri_params: FriParameters) -> Self {
        Self { fri_params }
    }
}
