use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig},
    system::SystemExecutor,
};
use openvm_circuit_derive::{InsExecutorE1, InstructionExecutor, VmConfig};
use openvm_stark_backend::{config::StarkGenericConfig, engine::StarkEngine, p3_field::Field};
use serde::{Deserialize, Serialize};

pub mod adapters;
mod auipc;
mod base_alu;
mod branch_eq;
mod branch_lt;
mod divrem;
mod hintstore;
mod jal_lui;
mod jalr;
mod less_than;
mod load_sign_extend;
mod loadstore;
mod mul;
mod mulh;
mod shift;

pub use auipc::*;
pub use base_alu::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use divrem::*;
pub use hintstore::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
pub use mul::*;
pub use mulh::*;
pub use shift::*;

mod extension;
pub use extension::*;

#[cfg(any(test, feature = "test-utils"))]
mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct Rv32IConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32IConfig {}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32ImConfig {
    #[config]
    pub rv32i: Rv32IConfig,
    #[extension]
    pub mul: Rv32M,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32ImConfig {}

impl Default for Rv32IConfig {
    fn default() -> Self {
        let system = SystemConfig::default().with_continuations();
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv32IConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        let system = SystemConfig::default()
            .with_continuations()
            .with_public_values(public_values)
            .with_max_segment_len(segment_len);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv32ImConfig {
    pub fn with_public_values(public_values: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values(public_values),
            mul: Default::default(),
        }
    }

    pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
        Self {
            rv32i: Rv32IConfig::with_public_values_and_segment_len(public_values, segment_len),
            mul: Default::default(),
        }
    }
}
