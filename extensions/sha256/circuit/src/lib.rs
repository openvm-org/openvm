#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]

use std::result::Result;

use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::VmConfig;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_stark_sdk::engine::StarkEngine;
use serde::{Deserialize, Serialize};

mod sha256_chip;
pub use sha256_chip::*;

mod extension;
pub use extension::*;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        use openvm_rv32im_circuit::Rv32ImGpuProverExt;
        pub(crate) mod cuda_abi;
        pub use Sha256Rv32GpuBuilder as Sha256Rv32Builder;
    } else {
        pub use Sha256Rv32CpuBuilder as Sha256Rv32Builder;
    }
}

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Sha256Rv32Config {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub sha256: Sha256,
}

impl Default for Sha256Rv32Config {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            sha256: Sha256,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Sha256Rv32Config {}

#[derive(Clone)]
pub struct Sha256Rv32CpuBuilder;

impl<E, SC> VmBuilder<E> for Sha256Rv32CpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = Sha256Rv32Config;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Sha256Rv32Config,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, &config.sha256, inventory)?;
        Ok(chip_complex)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Sha256Rv32GpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for Sha256Rv32GpuBuilder {
    type VmConfig = Sha256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Sha256Rv32Config,
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
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.rv32i,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.rv32m,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.io,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Sha256GpuProverExt,
            &config.sha256,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
