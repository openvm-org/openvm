use std::cmp::Ordering;

use openvm_circuit::arch::SystemConfig;

pub fn check_max_constraint_degrees(config: &SystemConfig, max_constraint_degree: usize) {
    match config.max_constraint_degree.cmp(&max_constraint_degree) {
        Ordering::Greater => {
            tracing::warn!(
                "config.max_constraint_degree ({}) > vk max_constraint_degree() ({})",
                config.max_constraint_degree,
                max_constraint_degree
            );
        }
        Ordering::Less => {
            tracing::info!(
                "config.max_constraint_degree ({}) < vk max_constraint_degree() ({})",
                config.max_constraint_degree,
                max_constraint_degree
            );
        }
        Ordering::Equal => {}
    }
}
