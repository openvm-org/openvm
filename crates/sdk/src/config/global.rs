use bon::Builder;
use openvm_algebra_circuit::{
    AlgebraCpuProverExt, Fp2Extension, Fp2ExtensionExecutor, ModularExtension,
    ModularExtensionExecutor,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::{Int256, Int256CpuProverExt, Int256Executor};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{instructions::NATIVE_AS, *},
    derive::VmConfig,
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_ecc_circuit::{
    EccCpuProverExt, WeierstrassExtension, WeierstrassExtensionExecutor, P256_CONFIG,
    SECP256K1_CONFIG,
};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256CpuProverExt, Keccak256Executor};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_native_circuit::{
    CastFExtension, CastFExtensionExecutor, Native, NativeCpuProverExt, NativeExecutor,
};
use openvm_native_transpiler::LongFormTranspilerExtension;
use openvm_pairing_circuit::{
    PairingCurve, PairingExtension, PairingExtensionExecutor, PairingProverExt,
    BLS12_381_COMPLEX_STRUCT_NAME, BN254_COMPLEX_STRUCT_NAME,
};
use openvm_pairing_transpiler::PairingTranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha2CpuProverExt};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::{Field, PrimeField32},
    prover::cpu::{CpuBackend, CpuDevice},
};
use openvm_transpiler::transpiler::Transpiler;
use serde::{Deserialize, Serialize};
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_algebra_circuit::AlgebraProverExt;
        use openvm_bigint_circuit::Int256GpuProverExt;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{
            engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend, types::SC,
        };
        use openvm_ecc_circuit::EccProverExt;
        use openvm_keccak256_circuit::Keccak256GpuProverExt;
        use openvm_native_circuit::NativeGpuProverExt;
        use openvm_rv32im_circuit::Rv32ImGpuProverExt;
        use openvm_sha256_circuit::Sha256GpuProverExt;
        pub use SdkVmGpuBuilder as SdkVmBuilder;
    } else {
        pub use SdkVmCpuBuilder as SdkVmBuilder;
    }
}

use super::AppFriParams;
use crate::{
    config::{AppConfig, TranspilerConfig},
    F,
};

/// The recommended way to construct [SdkVmConfig] is using [SdkVmConfig::from_toml].
///
/// For construction without reliance on deserialization, you can use [SdkVmConfigBuilder], which
/// follows a builder pattern. After calling [SdkVmConfigBuilder::build], call
/// [SdkVmConfig::optimize] to apply some default optimizations to built configuration for best
/// performance.
#[derive(Builder, Clone, Debug, Serialize, Deserialize)]
#[serde(from = "SdkVmConfigWithDefaultDeser")]
pub struct SdkVmConfig {
    pub system: SdkSystemConfig,
    pub rv32i: Option<UnitStruct>,
    pub io: Option<UnitStruct>,
    pub keccak: Option<UnitStruct>,
    pub sha256: Option<UnitStruct>,
    pub native: Option<UnitStruct>,
    pub castf: Option<UnitStruct>,

    /// NOTE: if enabling this together with the [Int256] extension, you should set the `rv32m`
    /// field to have the same `range_tuple_checker_sizes` as the `bigint` field for best
    /// performance.
    pub rv32m: Option<Rv32M>,
    /// NOTE: if enabling this together with the [Rv32M] extension, you should set the `rv32m`
    /// field to have the same `range_tuple_checker_sizes` as the `bigint` field for best
    /// performance.
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,
}

impl SdkVmConfig {
    /// Standard configuration with a set of default VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must match, including the order of
    /// the moduli and elliptic curve parameters of the respective extensions:
    /// The `app_vm_config` field of your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("openvm_standard.toml")]
    /// ```
    pub fn standard() -> SdkVmConfig {
        let bn_config = PairingCurve::Bn254.curve_config();
        let bls_config = PairingCurve::Bls12_381.curve_config();
        SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .keccak(Default::default())
            .sha256(Default::default())
            .bigint(Default::default())
            .modular(ModularExtension::new(vec![
                bn_config.modulus.clone(),
                bn_config.scalar.clone(),
                SECP256K1_CONFIG.modulus.clone(),
                SECP256K1_CONFIG.scalar.clone(),
                P256_CONFIG.modulus.clone(),
                P256_CONFIG.scalar.clone(),
                bls_config.modulus.clone(),
                bls_config.scalar.clone(),
            ]))
            .fp2(Fp2Extension::new(vec![
                (
                    BN254_COMPLEX_STRUCT_NAME.to_string(),
                    bn_config.modulus.clone(),
                ),
                (
                    BLS12_381_COMPLEX_STRUCT_NAME.to_string(),
                    bls_config.modulus.clone(),
                ),
            ]))
            .ecc(WeierstrassExtension::new(vec![
                bn_config.clone(),
                SECP256K1_CONFIG.clone(),
                P256_CONFIG.clone(),
                bls_config.clone(),
            ]))
            .pairing(PairingExtension::new(vec![
                PairingCurve::Bn254,
                PairingCurve::Bls12_381,
            ]))
            .build()
            .optimize()
    }

    /// Configuration with RISC-V RV32IM and IO VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("openvm_riscv32.toml")]
    /// ```
    pub fn riscv32() -> Self {
        SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .build()
            .optimize()
    }

    /// `openvm_toml` should be the TOML string read from an openvm.toml file.
    pub fn from_toml(openvm_toml: &str) -> Result<AppConfig<Self>, toml::de::Error> {
        toml::from_str(openvm_toml)
    }
}

impl AppConfig<SdkVmConfig> {
    pub fn standard() -> Self {
        Self::new(AppFriParams::default().fri_params, SdkVmConfig::standard())
    }

    pub fn riscv32() -> Self {
        Self::new(AppFriParams::default().fri_params, SdkVmConfig::riscv32())
    }
}

impl TranspilerConfig<F> for SdkVmConfig {
    fn transpiler(&self) -> Transpiler<F> {
        let mut transpiler = Transpiler::default();
        if self.rv32i.is_some() {
            transpiler = transpiler.with_extension(Rv32ITranspilerExtension);
        }
        if self.io.is_some() {
            transpiler = transpiler.with_extension(Rv32IoTranspilerExtension);
        }
        if self.keccak.is_some() {
            transpiler = transpiler.with_extension(Keccak256TranspilerExtension);
        }
        if self.sha256.is_some() {
            transpiler = transpiler.with_extension(Sha256TranspilerExtension);
        }
        if self.native.is_some() {
            transpiler = transpiler.with_extension(LongFormTranspilerExtension);
        }
        if self.rv32m.is_some() {
            transpiler = transpiler.with_extension(Rv32MTranspilerExtension);
        }
        if self.bigint.is_some() {
            transpiler = transpiler.with_extension(Int256TranspilerExtension);
        }
        if self.modular.is_some() {
            transpiler = transpiler.with_extension(ModularTranspilerExtension);
        }
        if self.fp2.is_some() {
            transpiler = transpiler.with_extension(Fp2TranspilerExtension);
        }
        if self.pairing.is_some() {
            transpiler = transpiler.with_extension(PairingTranspilerExtension);
        }
        if self.ecc.is_some() {
            transpiler = transpiler.with_extension(EccTranspilerExtension);
        }
        transpiler
    }
}

impl AsRef<SystemConfig> for SdkVmConfig {
    fn as_ref(&self) -> &SystemConfig {
        &self.system.config
    }
}

impl AsMut<SystemConfig> for SdkVmConfig {
    fn as_mut(&mut self) -> &mut SystemConfig {
        &mut self.system.config
    }
}

impl SdkVmConfig {
    pub fn optimize(mut self) -> Self {
        self.apply_optimizations();
        self
    }

    /// Apply small optimizations to the configuration.
    pub fn apply_optimizations(&mut self) {
        if self.native.is_none() && self.castf.is_none() {
            // There should be no need to write to native address space if Native extension and
            // CastF extension are not enabled.
            self.system.config.memory_config.addr_spaces[NATIVE_AS as usize].num_cells = 0;
        }
        let rv32m = self.rv32m.as_mut();
        let bigint = self.bigint.as_mut();
        if let (Some(bigint), Some(rv32m)) = (bigint, rv32m) {
            rv32m.range_tuple_checker_sizes[0] =
                rv32m.range_tuple_checker_sizes[0].max(bigint.range_tuple_checker_sizes[0]);
            rv32m.range_tuple_checker_sizes[1] =
                rv32m.range_tuple_checker_sizes[1].max(bigint.range_tuple_checker_sizes[1]);
            bigint.range_tuple_checker_sizes = rv32m.range_tuple_checker_sizes;
        }
    }

    pub fn to_inner(&self) -> SdkVmConfigInner {
        let config = self.clone().optimize();
        let system = config.system.config.clone();
        let rv32i = config.rv32i.map(|_| Rv32I);
        let io = config.io.map(|_| Rv32Io);
        let keccak = config.keccak.map(|_| Keccak256);
        let sha256 = config.sha256.map(|_| Sha256);
        let native = config.native.map(|_| Native);
        let castf = config.castf.map(|_| CastFExtension);
        let rv32m = config.rv32m;
        let bigint = config.bigint;
        let modular = config.modular.clone();
        let fp2 = config.fp2.clone();
        let pairing = config.pairing.clone();
        let ecc = config.ecc.clone();

        SdkVmConfigInner {
            system,
            rv32i,
            io,
            keccak,
            sha256,
            native,
            castf,
            rv32m,
            bigint,
            modular,
            fp2,
            pairing,
            ecc,
        }
    }
}

// ======================= Implementation of VmConfig and VmBuilder ====================

/// SDK CPU VmBuilder
#[derive(Copy, Clone, Default)]
pub struct SdkVmCpuBuilder;

/// Internal struct to use for the VmConfig derive macro.
/// Can be obtained via [`SdkVmConfig::to_inner`].
#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct SdkVmConfigInner {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension(executor = "Rv32IExecutor")]
    pub rv32i: Option<Rv32I>,
    #[extension(executor = "Rv32IoExecutor")]
    pub io: Option<Rv32Io>,
    #[extension(executor = "Keccak256Executor")]
    pub keccak: Option<Keccak256>,
    #[extension(executor = "Sha256Executor")]
    pub sha256: Option<Sha256>,
    #[extension(executor = "NativeExecutor<F>")]
    pub native: Option<Native>,
    #[extension(executor = "CastFExtensionExecutor")]
    pub castf: Option<CastFExtension>,

    #[extension(executor = "Rv32MExecutor")]
    pub rv32m: Option<Rv32M>,
    #[extension(executor = "Int256Executor")]
    pub bigint: Option<Int256>,
    #[extension(executor = "ModularExtensionExecutor")]
    pub modular: Option<ModularExtension>,
    #[extension(executor = "Fp2ExtensionExecutor")]
    pub fp2: Option<Fp2Extension>,
    #[extension(executor = "PairingExtensionExecutor<F>")]
    pub pairing: Option<PairingExtension>,
    #[extension(executor = "WeierstrassExtensionExecutor")]
    pub ecc: Option<WeierstrassExtension>,
}

// Generated by macro
pub type SdkVmConfigExecutor<F> = SdkVmConfigInnerExecutor<F>;

impl<F: Field> VmExecutionConfig<F> for SdkVmConfig
where
    SdkVmConfigInner: VmExecutionConfig<F>,
{
    type Executor = <SdkVmConfigInner as VmExecutionConfig<F>>::Executor;

    fn create_executors(
        &self,
    ) -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError> {
        self.to_inner().create_executors()
    }
}

impl<SC: StarkGenericConfig> VmCircuitConfig<SC> for SdkVmConfig
where
    SdkVmConfigInner: VmCircuitConfig<SC>,
{
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError> {
        self.to_inner().create_airs()
    }
}

impl<E, SC> VmBuilder<E> for SdkVmCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = SdkVmConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &SdkVmConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let config = config.to_inner();
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        if let Some(rv32i) = &config.rv32i {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, rv32i, inventory)?;
        }
        if let Some(io) = &config.io {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, io, inventory)?;
        }
        if let Some(keccak) = &config.keccak {
            VmProverExtension::<E, _, _>::extend_prover(&Keccak256CpuProverExt, keccak, inventory)?;
        }
        if let Some(sha256) = &config.sha256 {
            VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, sha256, inventory)?;
        }
        if let Some(native) = &config.native {
            VmProverExtension::<E, _, _>::extend_prover(&NativeCpuProverExt, native, inventory)?;
        }
        if let Some(castf) = &config.castf {
            VmProverExtension::<E, _, _>::extend_prover(&NativeCpuProverExt, castf, inventory)?;
        }
        if let Some(rv32m) = &config.rv32m {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, rv32m, inventory)?;
        }
        if let Some(bigint) = &config.bigint {
            VmProverExtension::<E, _, _>::extend_prover(&Int256CpuProverExt, bigint, inventory)?;
        }
        if let Some(modular) = &config.modular {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraCpuProverExt, modular, inventory)?;
        }
        if let Some(fp2) = &config.fp2 {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraCpuProverExt, fp2, inventory)?;
        }
        if let Some(pairing) = &config.pairing {
            VmProverExtension::<E, _, _>::extend_prover(&PairingProverExt, pairing, inventory)?;
        }
        if let Some(ecc) = &config.ecc {
            VmProverExtension::<E, _, _>::extend_prover(&EccCpuProverExt, ecc, inventory)?;
        }
        Ok(chip_complex)
    }
}

#[cfg(feature = "cuda")]
#[derive(Copy, Clone, Default)]
pub struct SdkVmGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for SdkVmGpuBuilder {
    type VmConfig = SdkVmConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &SdkVmConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        type E = GpuBabyBearPoseidon2Engine;

        let config = config.to_inner();
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemGpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        if let Some(rv32i) = &config.rv32i {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, rv32i, inventory)?;
        }
        if let Some(io) = &config.io {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, io, inventory)?;
        }
        if let Some(keccak) = &config.keccak {
            VmProverExtension::<E, _, _>::extend_prover(&Keccak256GpuProverExt, keccak, inventory)?;
        }
        if let Some(sha256) = &config.sha256 {
            VmProverExtension::<E, _, _>::extend_prover(&Sha256GpuProverExt, sha256, inventory)?;
        }
        if let Some(native) = &config.native {
            VmProverExtension::<E, _, _>::extend_prover(&NativeGpuProverExt, native, inventory)?;
        }
        if let Some(castf) = &config.castf {
            VmProverExtension::<E, _, _>::extend_prover(&NativeGpuProverExt, castf, inventory)?;
        }
        if let Some(rv32m) = &config.rv32m {
            VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, rv32m, inventory)?;
        }
        if let Some(bigint) = &config.bigint {
            VmProverExtension::<E, _, _>::extend_prover(&Int256GpuProverExt, bigint, inventory)?;
        }
        if let Some(modular) = &config.modular {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraProverExt, modular, inventory)?;
        }
        if let Some(fp2) = &config.fp2 {
            VmProverExtension::<E, _, _>::extend_prover(&AlgebraProverExt, fp2, inventory)?;
        }
        if let Some(pairing) = &config.pairing {
            VmProverExtension::<E, _, _>::extend_prover(&PairingProverExt, pairing, inventory)?;
        }
        if let Some(ecc) = &config.ecc {
            VmProverExtension::<E, _, _>::extend_prover(&EccProverExt, ecc, inventory)?;
        }
        Ok(chip_complex)
    }
}

// ======================= Boilerplate ====================

impl InitFileGenerator for SdkVmConfig {
    fn generate_init_file_contents(&self) -> Option<String> {
        self.to_inner().generate_init_file_contents()
    }
}
impl InitFileGenerator for SdkVmConfigInner {
    fn generate_init_file_contents(&self) -> Option<String> {
        if self.modular.is_some() || self.fp2.is_some() || self.ecc.is_some() {
            let mut contents = String::new();
            contents.push_str(
                "// This file is automatically generated by cargo openvm. Do not rename or edit.\n",
            );

            if let Some(modular_config) = &self.modular {
                contents.push_str(&modular_config.generate_moduli_init());
                contents.push('\n');
            }

            if let Some(fp2_config) = &self.fp2 {
                assert!(
                    self.modular.is_some(),
                    "ModularExtension is required for Fp2Extension"
                );
                let modular_config = self.modular.as_ref().unwrap();
                contents.push_str(&fp2_config.generate_complex_init(modular_config));
                contents.push('\n');
            }

            if let Some(ecc_config) = &self.ecc {
                contents.push_str(&ecc_config.generate_sw_init());
                contents.push('\n');
            }

            Some(contents)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SdkSystemConfig {
    pub config: SystemConfig,
}

// Default implementation uses no init file
impl InitFileGenerator for SdkSystemConfig {}

impl From<SystemConfig> for SdkSystemConfig {
    fn from(config: SystemConfig) -> Self {
        Self { config }
    }
}

/// A struct that is used to represent a unit struct in the config, used for
/// serialization and deserialization.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct UnitStruct {}

impl From<Rv32I> for UnitStruct {
    fn from(_: Rv32I) -> Self {
        UnitStruct {}
    }
}

impl From<Rv32Io> for UnitStruct {
    fn from(_: Rv32Io) -> Self {
        UnitStruct {}
    }
}

impl From<Keccak256> for UnitStruct {
    fn from(_: Keccak256) -> Self {
        UnitStruct {}
    }
}

impl From<Sha256> for UnitStruct {
    fn from(_: Sha256) -> Self {
        UnitStruct {}
    }
}

impl From<Native> for UnitStruct {
    fn from(_: Native) -> Self {
        UnitStruct {}
    }
}

impl From<CastFExtension> for UnitStruct {
    fn from(_: CastFExtension) -> Self {
        UnitStruct {}
    }
}

#[derive(Deserialize)]
struct SdkVmConfigWithDefaultDeser {
    #[serde(default)]
    pub system: SdkSystemConfig,

    pub rv32i: Option<UnitStruct>,
    pub io: Option<UnitStruct>,
    pub keccak: Option<UnitStruct>,
    pub sha256: Option<UnitStruct>,
    pub native: Option<UnitStruct>,
    pub castf: Option<UnitStruct>,

    pub rv32m: Option<Rv32M>,
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,
}

impl From<SdkVmConfigWithDefaultDeser> for SdkVmConfig {
    fn from(config: SdkVmConfigWithDefaultDeser) -> Self {
        let ret = Self {
            system: config.system,
            rv32i: config.rv32i,
            io: config.io,
            keccak: config.keccak,
            sha256: config.sha256,
            native: config.native,
            castf: config.castf,
            rv32m: config.rv32m,
            bigint: config.bigint,
            modular: config.modular,
            fp2: config.fp2,
            pairing: config.pairing,
            ecc: config.ecc,
        };
        ret.optimize()
    }
}

#[cfg(test)]
mod tests {
    use itertools::zip_eq;

    use super::*;

    #[test]
    fn test_app_config_consistency() {
        let toml_config = SdkVmConfig::from_toml(include_str!("./openvm_standard.toml")).unwrap();
        for (line1, line2) in zip_eq(
            toml::to_string_pretty(&AppConfig::standard())
                .unwrap()
                .lines(),
            toml::to_string_pretty(&toml_config).unwrap().lines(),
        ) {
            assert_eq!(line1, line2);
        }

        let toml_config = SdkVmConfig::from_toml(include_str!("./openvm_riscv32.toml")).unwrap();
        for (line1, line2) in zip_eq(
            toml::to_string_pretty(&AppConfig::riscv32())
                .unwrap()
                .lines(),
            toml::to_string_pretty(&toml_config).unwrap().lines(),
        ) {
            assert_eq!(line1, line2);
        }
    }
}
