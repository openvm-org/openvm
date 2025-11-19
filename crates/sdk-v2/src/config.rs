use clap::Args;
use openvm_sdk_config::SdkVmConfig;
use serde::{Deserialize, Serialize};
use stark_backend_v2::SystemParams;

// Aggregation Tree Defaults
pub const DEFAULT_NUM_CHILDREN_LEAF: usize = 1;
pub const DEFAULT_NUM_CHILDREN_INTERNAL: usize = 3;

pub const DEFAULT_LEAF_PARAMS: SystemParams = SystemParams {
    l_skip: 2,
    n_stack: 17,
    log_blowup: 3,
    k_whir: 4,
    num_whir_queries: 30,
    log_final_poly_len: 7,
    logup_pow_bits: 16,
    whir_pow_bits: 16,
};
pub const DEFAULT_INTERNAL_PARAMS: SystemParams = SystemParams {
    l_skip: 2,
    n_stack: 17,
    log_blowup: 3,
    k_whir: 4,
    num_whir_queries: 30,
    log_final_poly_len: 7,
    logup_pow_bits: 16,
    whir_pow_bits: 16,
};

#[derive(Clone, Debug, Serialize, Deserialize, derive_new::new)]
pub struct AppConfig<VC> {
    pub app_vm_config: VC,
    pub system_params: SystemParams,
}

impl AppConfig<SdkVmConfig> {
    pub fn standard(params: SystemParams) -> Self {
        Self::new(SdkVmConfig::standard(), params)
    }

    pub fn riscv32(params: SystemParams) -> Self {
        Self::new(SdkVmConfig::riscv32(), params)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub max_num_user_public_values: usize,
    pub params: AggregationSystemParams,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AggregationSystemParams {
    pub leaf: SystemParams,
    pub internal: SystemParams,
}

impl Default for AggregationSystemParams {
    fn default() -> Self {
        Self {
            leaf: DEFAULT_LEAF_PARAMS,
            internal: DEFAULT_INTERNAL_PARAMS,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Args)]
pub struct AggregationTreeConfig {
    /// Each leaf verifier circuit will aggregate this many App VM proofs.
    #[arg(
        long,
        default_value_t = DEFAULT_NUM_CHILDREN_LEAF,
        help = "Number of children per leaf verifier circuit",
        help_heading = "Aggregation Tree Options"
    )]
    pub num_children_leaf: usize,
    /// Each internal verifier circuit will aggregate this many proofs,
    /// where each proof may be of either leaf or internal verifier (self) circuit.
    #[arg(
        long,
        default_value_t = DEFAULT_NUM_CHILDREN_INTERNAL,
        help = "Number of children per internal verifier circuit",
        help_heading = "Aggregation Tree Options"
    )]
    pub num_children_internal: usize,
}

impl Default for AggregationTreeConfig {
    fn default() -> Self {
        Self {
            num_children_leaf: DEFAULT_NUM_CHILDREN_LEAF,
            num_children_internal: DEFAULT_NUM_CHILDREN_INTERNAL,
        }
    }
}
