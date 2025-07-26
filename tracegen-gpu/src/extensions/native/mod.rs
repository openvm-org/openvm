mod branch_eq;
mod castf;
mod cuda;
mod extension;
mod field_arithmetic;
mod field_extension;
mod fri;
mod jal_rangecheck;
mod loadstore;
mod poseidon2;

pub use branch_eq::*;
pub use castf::*;
pub use cuda::*;
pub use extension::*;
pub use field_arithmetic::*;
pub use field_extension::*;
pub use fri::*;
pub use jal_rangecheck::*;
pub use loadstore::*;
pub use poseidon2::*;

mod utils;
use openvm_circuit::arch::{
    AirInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex, VmProverExtension,
};
use openvm_native_circuit::{NativeConfig, Rv32WithKernelsConfig};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
#[cfg(any(test, feature = "test-utils"))]
pub use utils::test_utils::*;

use crate::{
    extensions::rv32im::Rv32ImGpuProverExt,
    system::{extensions::SystemGpuBuilder, SystemChipInventoryGPU},
};

type E = GpuBabyBearPoseidon2Engine;

#[derive(Clone)]
pub struct NativeGpuBuilder;

impl VmBuilder<E> for NativeGpuBuilder {
    type VmConfig = NativeConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &NativeConfig,
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
        VmProverExtension::<E, _, _>::extend_prover(
            &NativeGpuProverExt,
            &config.native,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[derive(Clone)]
pub struct Rv32WithKernelsGpuBuilder;

impl VmBuilder<E> for Rv32WithKernelsGpuBuilder {
    type VmConfig = Rv32WithKernelsConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32WithKernelsConfig,
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
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &NativeGpuProverExt,
            &config.native,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(&NativeGpuProverExt, &config.castf, inventory)?;
        Ok(chip_complex)
    }
}
