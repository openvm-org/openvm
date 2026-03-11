use clap::Args;
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_backend::SystemParams;
use openvm_stark_sdk::config::{
    internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
};
pub use openvm_stark_sdk::config::{
    DEFAULT_APP_LOG_BLOWUP, DEFAULT_APP_L_SKIP, DEFAULT_INTERNAL_LOG_BLOWUP,
    DEFAULT_LEAF_LOG_BLOWUP, DEFAULT_ROOT_LOG_BLOWUP,
};
use serde::{Deserialize, Serialize};

// WARNING: These currently serve as both the DEFAULT and MAXIMUM number of
// children for the leaf and internal aggregation layers, as the max number
// of children is a const generic in the recursion circuit. We may change
// these as needed, but note that a disparity in max and actual number of
// leaf/internal children will cause a performance loss.
pub const MAX_NUM_CHILDREN_LEAF: usize = 4;
pub const MAX_NUM_CHILDREN_INTERNAL: usize = 3;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationConfig {
    pub params: AggregationSystemParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregationSystemParams {
    pub leaf: SystemParams,
    pub internal: SystemParams,
}

impl Default for AggregationSystemParams {
    fn default() -> Self {
        Self {
            leaf: leaf_params_with_100_bits_security(),
            internal: internal_params_with_100_bits_security(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Args)]
pub struct AggregationTreeConfig {
    /// Each leaf verifier circuit will aggregate this many App VM proofs.
    #[arg(
        long,
        default_value_t = MAX_NUM_CHILDREN_LEAF,
        help = "Number of children per leaf verifier circuit",
        help_heading = "Aggregation Tree Options"
    )]
    pub num_children_leaf: usize,
    /// Each internal verifier circuit will aggregate this many proofs,
    /// where each proof may be of either leaf or internal verifier (self) circuit.
    #[arg(
        long,
        default_value_t = MAX_NUM_CHILDREN_INTERNAL,
        help = "Number of children per internal verifier circuit",
        help_heading = "Aggregation Tree Options"
    )]
    pub num_children_internal: usize,
}

impl Default for AggregationTreeConfig {
    fn default() -> Self {
        Self {
            num_children_leaf: MAX_NUM_CHILDREN_LEAF,
            num_children_internal: MAX_NUM_CHILDREN_INTERNAL,
        }
    }
}
