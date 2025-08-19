use openvm_circuit::arch::SystemConfig;
use openvm_stark_sdk::config::FriParameters;

pub fn check_max_constraint_degrees(config: &SystemConfig, fri_params: &FriParameters) {
    if config.max_constraint_degree != fri_params.max_constraint_degree() {
        tracing::warn!(
            "config.max_constraint_degree ({}) != fri_params.max_constraint_degree() ({})",
            config.max_constraint_degree,
            fri_params.max_constraint_degree()
        );
    }
}
