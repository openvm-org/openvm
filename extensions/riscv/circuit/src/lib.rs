#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, SystemConfig, VmBuilder,
        VmChipComplex, VmField, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{Executor, MeteredExecutor, VmConfig};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_stark_backend::{EngineDeviceCtx, StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

pub mod adapters;
mod add_sub;
mod add_sub_w;
mod addi;
mod auipc;
mod bitwise_logic;
mod bitwise_logic_imm;
mod branch_eq;
mod branch_lt;
mod divrem;
mod divrem_w;
mod hintstore;
mod jal_lui;
mod jalr;
mod less_than;
mod less_than_imm;
mod load;
mod load_sign_extend;
mod mul;
mod mul_w;
mod mulh;
mod shift_logical;
mod shift_logical_imm;
mod shift_right_arithmetic;
mod shift_right_arithmetic_imm;
mod shift_w;
mod store;

pub use add_sub::*;
pub use add_sub_w::*;
pub use addi::*;
pub use auipc::*;
pub use bitwise_logic::*;
pub use bitwise_logic_imm::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use divrem::*;
pub use divrem_w::*;
pub use hintstore::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use less_than_imm::*;
pub use load::*;
pub use load_sign_extend::*;
pub use mul::*;
pub use mul_w::*;
pub use mulh::*;
pub use shift_logical::*;
pub use shift_logical_imm::*;
pub use shift_right_arithmetic::*;
pub use shift_right_arithmetic_imm::*;
pub use shift_w::*;
pub use store::*;

mod extension;
pub use extension::*;

#[cfg(all(feature = "cuda", feature = "rvr"))]
pub mod preflight;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        pub(crate) mod cuda_abi;
        pub use self::{
            RiscvIGpuBuilder as RiscvIBuilder,
            RiscvImGpuBuilder as RiscvImBuilder,
        };
    } else {
        pub use self::{
            RiscvICpuBuilder as RiscvIBuilder,
            RiscvImCpuBuilder as RiscvImBuilder,
        };
    }
}

#[cfg(any(test, feature = "test-utils"))]
mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct RiscvIConfig {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub base: RiscvI,
    #[extension]
    pub io: RiscvIo,
}

// Default implementation uses no init file
impl InitFileGenerator for RiscvIConfig {}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct RiscvImConfig {
    #[config]
    pub riscv_i: RiscvIConfig,
    #[extension]
    pub mul: RiscvM,
}

// Default implementation uses no init file
impl InitFileGenerator for RiscvImConfig {}

impl Default for RiscvIConfig {
    fn default() -> Self {
        let system = SystemConfig::default();
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl RiscvIConfig {
    pub fn with_public_values_bytes(num_public_values_bytes: usize) -> Self {
        let system = SystemConfig::default().with_public_values_bytes(num_public_values_bytes);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl RiscvImConfig {
    pub fn with_public_values_bytes(num_public_values_bytes: usize) -> Self {
        Self {
            riscv_i: RiscvIConfig::with_public_values_bytes(num_public_values_bytes),
            mul: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct RiscvICpuBuilder;

impl<SC, E> VmBuilder<E> for RiscvICpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = RiscvIConfig;
    type SystemChipInventory = SystemChipInventory<SC>;

    fn create_chip_complex(
        &self,
        config: &RiscvIConfig,
        circuit: AirInventory<E::SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<VmChipComplex<E::SC, E::PB, Self::SystemChipInventory>, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.io, inventory)?;
        Ok(chip_complex)
    }
}

#[derive(Clone)]
pub struct RiscvImCpuBuilder;

impl<SC, E> VmBuilder<E> for RiscvImCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = RiscvImConfig;
    type SystemChipInventory = SystemChipInventory<SC>;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<E::SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<VmChipComplex<E::SC, E::PB, Self::SystemChipInventory>, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &RiscvICpuBuilder,
            &config.riscv_i,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(&RiscvImCpuProverExt, &config.mul, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct RiscvIGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for RiscvIGpuBuilder {
    type VmConfig = RiscvIConfig;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &RiscvIConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &EngineDeviceCtx<GpuBabyBearPoseidon2Engine>,
    ) -> Result<
        VmChipComplex<BabyBearPoseidon2Config, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _>::extend_prover(
            &RiscvImGpuProverExt,
            &config.base,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _>::extend_prover(
            &RiscvImGpuProverExt,
            &config.io,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct RiscvImGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for RiscvImGpuBuilder {
    type VmConfig = RiscvImConfig;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &EngineDeviceCtx<GpuBabyBearPoseidon2Engine>,
    ) -> Result<
        VmChipComplex<BabyBearPoseidon2Config, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &RiscvIGpuBuilder,
            &config.riscv_i,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _>::extend_prover(
            &RiscvImGpuProverExt,
            &config.mul,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
