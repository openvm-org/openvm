#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]

use openvm_circuit::{
    arch::{InitFileGenerator, MemoryConfig, SystemConfig},
    system::SystemExecutor,
};
use openvm_circuit_derive::VmConfig;
use openvm_instructions::riscv::RV32_REGISTER_AS;
use serde::{Deserialize, Serialize};

mod auipc;
mod base_alu;
mod base_alu_w;
mod branch_eq;
mod branch_lt;
mod divrem;
mod divrem_w;
mod extension;
mod hintstore;
mod jal_lui;
mod jalr;
mod less_than;
mod load_sign_extend;
mod loadstore;
mod mul;
mod mul_w;
mod mulh;
mod shift;
mod shift_w;
#[cfg(test)]
mod test_utils;

pub use auipc::*;
pub use base_alu::*;
pub use base_alu_w::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use divrem::*;
pub use divrem_w::*;
pub use extension::*;
pub use hintstore::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
pub use mul::*;
pub use mul_w::*;
pub use mulh::*;
pub use shift::*;
pub use shift_w::*;

/// Memory configuration for RV64: 32 registers x 8 bytes = 256 bytes register space.
pub fn rv64_mem_config() -> MemoryConfig {
    let mut config = MemoryConfig::default();
    config.addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * 8;
    config
}

/// Config for a VM with RV64I base extension and IO extension.
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct Rv64IConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv64I,
    #[extension]
    pub io: Rv64Io,
}

impl InitFileGenerator for Rv64IConfig {}

impl Default for Rv64IConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default_from_memory(rv64_mem_config()),
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv64IConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system =
            SystemConfig::default_from_memory(rv64_mem_config()).with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = SystemConfig::default_from_memory(rv64_mem_config())
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

/// Config for a VM with RV64I base extension, IO extension, and M (multiplication) extension.
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv64ImConfig {
    #[config]
    pub rv64i: Rv64IConfig,
    #[extension]
    pub mul: Rv64M,
}

impl InitFileGenerator for Rv64ImConfig {}

impl Rv64ImConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        Self {
            rv64i: Rv64IConfig::with_public_values(public_values),
            mul: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        Self {
            rv64i: Rv64IConfig::with_public_values_and_segment_len(public_values, segment_len),
            mul: Default::default(),
        }
    }
}
