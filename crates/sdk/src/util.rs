use std::cmp::Ordering;

use openvm_circuit::arch::SystemConfig;
use openvm_stark_sdk::config::FriParameters;

pub fn check_max_constraint_degrees(config: &SystemConfig, fri_params: &FriParameters) {
    match config
        .max_constraint_degree
        .cmp(&fri_params.max_constraint_degree())
    {
        Ordering::Greater => {
            tracing::warn!(
                "config.max_constraint_degree ({}) > fri_params.max_constraint_degree() ({})",
                config.max_constraint_degree,
                fri_params.max_constraint_degree()
            );
        }
        Ordering::Less => {
            tracing::info!(
                "config.max_constraint_degree ({}) < fri_params.max_constraint_degree() ({})",
                config.max_constraint_degree,
                fri_params.max_constraint_degree()
            );
        }
        Ordering::Equal => {}
    }
}
