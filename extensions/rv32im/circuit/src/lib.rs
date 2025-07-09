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
use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig},
    system::SystemExecutor,
};
use openvm_circuit_derive::{AnyEnum, InsExecutorE1, InstructionExecutor, VmConfig};
use openvm_stark_backend::p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
pub use shift::*;
use strum::IntoEnumIterator;

mod extension;
pub use extension::*;

// #[cfg(any(test, feature = "test-utils"))]
// mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, Serialize, Deserialize)]
pub struct Rv32IConfig {
    #[config(executor = SystemExecutor)]
    pub system: SystemConfig,
    #[extension(generics = false)]
    pub base: Rv32I,
    #[extension(generics = false)]
    pub io: Rv32Io,
}
#[derive(
    ::openvm_circuit::derive::InstructionExecutor,
    ::openvm_circuit::derive::InsExecutorE1,
    ::derive_more::derive::From,
    ::openvm_circuit::derive::AnyEnum,
)]
pub enum Rv32IConfigExecutor<F: PrimeField32> {
    #[any_enum]
    System(SystemExecutor<F>),
    #[any_enum]
    Base(Rv32IExecutor),
    #[any_enum]
    Io(Rv32IoExecutor),
}
impl<F: PrimeField32> ::openvm_circuit::arch::VmExecutionConfig<F> for Rv32IConfig {
    type Executor = Rv32IConfigExecutor<F>;
    fn create_executors(
        &self,
    ) -> Result<
        ::openvm_circuit::arch::ExecutorInventory<Self::Executor>,
        ::openvm_circuit::arch::ExecutorInventoryError,
    > {
        let inventory = self.system.create_executors()?;
        let inventory: ::openvm_circuit::arch::ExecutorInventory<Self::Executor> =
            inventory.extend(&self.base)?;
        let inventory: ::openvm_circuit::arch::ExecutorInventory<Self::Executor> =
            inventory.extend(&self.io)?;
        Ok(inventory)
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32IConfig {}

// /// Config for a VM with base extension, IO extension, and multiplication extension
// #[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
// pub struct Rv32ImConfig {
//     #[config]
//     pub rv32i: Rv32IConfig,
//     #[extension(generics = false)]
//     pub mul: Rv32M,
// }

// // Default implementation uses no init file
// impl InitFileGenerator for Rv32ImConfig {}

// impl Default for Rv32IConfig {
//     fn default() -> Self {
//         let system = SystemConfig::default().with_continuations();
//         Self {
//             system,
//             base: Default::default(),
//             io: Default::default(),
//         }
//     }
// }

// impl Rv32IConfig {
//     pub fn with_public_values(public_values: usize) -> Self {
//         let system = SystemConfig::default()
//             .with_continuations()
//             .with_public_values(public_values);
//         Self {
//             system,
//             base: Default::default(),
//             io: Default::default(),
//         }
//     }

//     pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
//         let system = SystemConfig::default()
//             .with_continuations()
//             .with_public_values(public_values)
//             .with_max_segment_len(segment_len);
//         Self {
//             system,
//             base: Default::default(),
//             io: Default::default(),
//         }
//     }
// }

// impl Rv32ImConfig {
//     pub fn with_public_values(public_values: usize) -> Self {
//         Self {
//             rv32i: Rv32IConfig::with_public_values(public_values),
//             mul: Default::default(),
//         }
//     }

//     pub fn with_public_values_and_segment_len(public_values: usize, segment_len: usize) -> Self {
//         Self {
//             rv32i: Rv32IConfig::with_public_values_and_segment_len(public_values, segment_len),
//             mul: Default::default(),
//         }
//     }
// }
