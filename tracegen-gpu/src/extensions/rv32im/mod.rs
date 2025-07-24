mod auipc;
mod base_alu;
mod branch_eq;
mod branch_lt;
mod cuda;
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
pub use cuda::*;
pub use divrem::*;
pub use hintstore::*;
pub use jal_lui::*;
pub use jalr::*;
pub use less_than::*;
pub use load_sign_extend::*;
pub use loadstore::*;
pub use mul::*;
pub use mulh::*;
use openvm_circuit::arch::{
    AirInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex, VmProverExtension,
};
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
pub use shift::*;

mod extension;
pub use extension::*;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};

use crate::system::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};

type E = GpuBabyBearPoseidon2Engine;
#[derive(Clone)]
pub struct Rv32IGpuBuilder;

#[derive(Clone)]
pub struct Rv32ImGpuBuilder;

impl VmBuilder<E> for Rv32IGpuBuilder {
    type VmConfig = Rv32IConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32IConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemGpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.io, inventory)?;
        Ok(chip_complex)
    }
}

impl VmBuilder<E> for Rv32ImGpuBuilder {
    type VmConfig = Rv32ImConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&Rv32IGpuBuilder, &config.rv32i, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.mul, inventory)?;
        Ok(chip_complex)
    }
}
