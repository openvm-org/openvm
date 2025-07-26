use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, MemoryConfig,
        SystemConfig, VmBuilder, VmChipComplex, VmProverExtension,
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

pub mod adapters;

mod branch_eq;
mod castf;
mod field_arithmetic;
mod field_extension;
mod fri;
mod jal_rangecheck;
mod loadstore;
mod poseidon2;

pub use branch_eq::*;
pub use castf::*;
pub use field_arithmetic::*;
pub use field_extension::*;
pub use fri::*;
pub use jal_rangecheck::*;
pub use loadstore::*;
pub use poseidon2::*;

mod extension;
pub use extension::*;

mod utils;
#[cfg(any(test, feature = "test-utils"))]
pub use utils::test_utils::*;
pub(crate) use utils::*;

#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct NativeConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension(generics = true)]
    pub native: Native,
}

impl NativeConfig {
    pub fn aggregation(num_public_values: usize, max_constraint_degree: usize) -> Self {
        Self {
            system: SystemConfig::new(
                max_constraint_degree,
                MemoryConfig::aggregation(),
                num_public_values,
            )
            .with_max_segment_len((1 << 24) - 100),
            native: Default::default(),
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for NativeConfig {}

#[derive(Clone, Default)]
pub struct NativeCpuBuilder;

impl<E, SC> VmBuilder<E> for NativeCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = NativeConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &NativeConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &NativeCpuProverExt,
            &config.native,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

#[derive(Clone, Debug, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv32WithKernelsConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension(generics = true)]
    pub native: Native,
    #[extension]
    pub castf: CastFExtension,
}

impl Default for Rv32WithKernelsConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            native: Native,
            castf: CastFExtension,
        }
    }
}

// Default implementation uses no init file
impl InitFileGenerator for Rv32WithKernelsConfig {}

#[derive(Clone)]
pub struct Rv32WithKernelsCpuBuilder;

impl<E, SC> VmBuilder<E> for Rv32WithKernelsCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = Rv32WithKernelsConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Rv32WithKernelsConfig,
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
        VmProverExtension::<E, _, _>::extend_prover(
            &NativeCpuProverExt,
            &config.native,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(&NativeCpuProverExt, &config.castf, inventory)?;
        Ok(chip_complex)
    }
}
