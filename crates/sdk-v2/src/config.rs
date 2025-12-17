use clap::Args;
use openvm_sdk_config::SdkVmConfig;
use openvm_stark_sdk::config::log_up_params::log_up_security_params_baby_bear_100_bits;
use serde::{Deserialize, Serialize};
use stark_backend_v2::{SystemParams, WhirConfig, WhirParams};

pub const DEFAULT_APP_L_SKIP: usize = 4;
pub const DEFAULT_APP_LOG_BLOWUP: usize = 1;
pub const DEFAULT_LEAF_LOG_BLOWUP: usize = 2;
pub const DEFAULT_INTERNAL_LOG_BLOWUP: usize = 2;

// WARNING: These currently serve as both the DEFAULT and MAXIMUM number of
// children for the leaf and internal aggregation layers, as the max number
// of children is a const generic in the recursion circuit. We may change
// these as needed, but note that a disparity in max and actual number of
// leaf/internal children will cause a performance loss.
pub const MAX_NUM_CHILDREN_LEAF: usize = 1;
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
    pub max_num_user_public_values: usize,
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
            leaf: default_leaf_params(DEFAULT_LEAF_LOG_BLOWUP),
            internal: default_internal_params(DEFAULT_INTERNAL_LOG_BLOWUP),
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

/// App params are configurable for max_log_height = l_skip + n_stack.
/// `l_skip` is tuned separately for performance.
/// `log_final_poly_len` is determined from `l_skip, n_stack` and adjusted in multiples of `k_whir`
/// to be <= 10;
pub fn default_app_params(log_blowup: usize, l_skip: usize, n_stack: usize) -> SystemParams {
    let k_whir = 4;
    let max_constraint_degree = 4;
    generic_system_params(log_blowup, l_skip, n_stack, k_whir, max_constraint_degree)
}

pub fn default_leaf_params(log_blowup: usize) -> SystemParams {
    let l_skip = 2;
    let n_stack = 17;
    let k_whir = 4;
    let max_constraint_degree = 4;
    generic_system_params(log_blowup, l_skip, n_stack, k_whir, max_constraint_degree)
}

pub fn default_internal_params(log_blowup: usize) -> SystemParams {
    let l_skip = 2;
    let n_stack = 17;
    let k_whir = 4;
    let max_constraint_degree = 4;
    generic_system_params(log_blowup, l_skip, n_stack, k_whir, max_constraint_degree)
}

pub fn generic_system_params(
    log_blowup: usize,
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    max_constraint_degree: usize,
) -> SystemParams {
    let whir_params = WhirParams {
        k: k_whir,
        log_final_poly_len: WHIR_MAX_LOG_FINAL_POLY_LEN,
        query_phase_pow_bits: WHIR_POW_BITS,
    };
    let whir_config = WhirConfig::new(log_blowup, l_skip + n_stack, whir_params, SECURITY_LEVEL);
    SystemParams {
        l_skip,
        n_stack,
        log_blowup,
        whir: whir_config,
        logup: log_up_security_params_baby_bear_100_bits(),
        max_constraint_degree,
    }
}

// TODO: move to stark-backend-v2
const WHIR_MAX_LOG_FINAL_POLY_LEN: usize = 10;
const WHIR_POW_BITS: usize = 20;
const SECURITY_LEVEL: usize = 100;
