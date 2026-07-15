use bon::Builder;
use openvm_algebra_circuit::*;
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::*;
use openvm_bigint_transpiler::*;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::{LogNativeAssemblerRegistry, VmRvrLogNativeExtension};
use openvm_circuit::{
    arch::{instructions::DEFERRAL_AS, *},
    derive::VmConfig,
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_deferral_circuit::*;
use openvm_deferral_transpiler::*;
use openvm_ecc_circuit::*;
use openvm_ecc_transpiler::*;
use openvm_keccak256_circuit::*;
use openvm_keccak256_transpiler::*;
use openvm_pairing_circuit::*;
use openvm_pairing_transpiler::*;
use openvm_riscv_circuit::*;
use openvm_riscv_transpiler::*;
use openvm_sha2_circuit::*;
use openvm_sha2_transpiler::*;
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::{p3_field::Field, StarkEngine, StarkProtocolConfig, Val};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use openvm_transpiler::transpiler::Transpiler;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::RvrExtensions;
use serde::{Deserialize, Serialize};

pub mod deferral;
use deferral::DeferralConfig;
#[cfg(feature = "rvr")]
use {
    openvm_algebra_circuit::log_native::ModularRecordArena,
    openvm_bigint_circuit::log_native::Int256RecordArena,
    openvm_ecc_circuit::log_native::WeierstrassRecordArena,
    openvm_keccak256_circuit::KeccakRecordArena,
    openvm_riscv_circuit::log_native::{Rv64IRecordArena, Rv64IoRecordArena, Rv64MRecordArena},
    openvm_sha2_circuit::Sha256RecordArena,
};
cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_algebra_circuit::AlgebraProverExt;
        use openvm_bigint_circuit::Int256GpuProverExt;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{
            BabyBearPoseidon2GpuEngine, GpuBackend
        };
        use openvm_ecc_circuit::EccProverExt;
        use openvm_keccak256_circuit::Keccak256GpuProverExt;
        use openvm_riscv_circuit::Rv64ImGpuProverExt;
        use openvm_sha2_circuit::Sha2GpuProverExt;
        pub use SdkVmGpuBuilder as SdkVmBuilder;
    } else {
        pub use SdkVmCpuBuilder as SdkVmBuilder;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SdkVmConfigWrapper {
    app_vm_config: SdkVmConfig,
}

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
    pub rv64i: Option<UnitStruct>,
    pub io: Option<UnitStruct>,
    pub keccak: Option<UnitStruct>,
    pub sha2: Option<UnitStruct>,

    /// NOTE: if enabling this together with the [Int256] extension, you should set the `rv64m`
    /// field to have the same `range_tuple_checker_sizes` as the `bigint` field for best
    /// performance.
    pub rv64m: Option<Rv64M>,
    /// NOTE: if enabling this together with the [Rv64M] extension, you should set the `rv64m`
    /// field to have the same `range_tuple_checker_sizes` as the `bigint` field for best
    /// performance.
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,

    /// NOTE: Only deferral configurations enumerated by SupportedDeferral are fully serializable
    /// and deserializable. For custom deferral circuits stored as SupportedDeferral::Other, the
    /// circuit's DeferralFn must be manually replaced in the extension after deserialization.
    pub deferral: Option<DeferralConfig>,
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
            .rv64i(Default::default())
            .rv64m(Default::default())
            .io(Default::default())
            .keccak(Default::default())
            .sha2(Default::default())
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

    /// Configuration with RISC-V RV64IM and IO VM extensions loaded.
    ///
    /// **Note**: To use this configuration, your `openvm.toml` must exactly match the following:
    ///
    /// ```toml
    #[doc = include_str!("openvm_riscv64.toml")]
    /// ```
    pub fn riscv64() -> Self {
        SdkVmConfig::builder()
            .system(Default::default())
            .rv64i(Default::default())
            .rv64m(Default::default())
            .io(Default::default())
            .build()
            .optimize()
    }

    /// `openvm_toml` should be the TOML string read from an openvm.toml file.
    pub fn from_toml(openvm_toml: &str) -> Result<Self, toml::de::Error> {
        let wrapper: SdkVmConfigWrapper = toml::from_str(openvm_toml)?;
        Ok(wrapper.app_vm_config)
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(&SdkVmConfigWrapper {
            app_vm_config: self.clone(),
        })
    }
}

pub trait TranspilerConfig<F> {
    fn transpiler(&self) -> Transpiler<F>;
}

impl TranspilerConfig<F> for SdkVmConfig {
    fn transpiler(&self) -> Transpiler<F> {
        let mut transpiler = Transpiler::default();
        if self.rv64i.is_some() {
            transpiler = transpiler.with_extension(Rv64ITranspilerExtension);
        }
        if self.io.is_some() {
            transpiler = transpiler.with_extension(Rv64IoTranspilerExtension);
        }
        if self.keccak.is_some() {
            transpiler = transpiler.with_extension(Keccak256TranspilerExtension);
        }
        if self.sha2.is_some() {
            transpiler = transpiler.with_extension(Sha2TranspilerExtension);
        }
        if self.rv64m.is_some() {
            transpiler = transpiler.with_extension(Rv64MTranspilerExtension);
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
        if let Some(deferral_config) = &self.deferral {
            transpiler = transpiler.with_extension(DeferralTranspilerExtension::new(
                deferral_config.to_extension().def_circuit_commits,
            ));
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
        let rv64m = self.rv64m.as_mut();
        let bigint = self.bigint.as_mut();
        if let (Some(bigint), Some(rv64m)) = (bigint, rv64m) {
            rv64m.range_tuple_checker_sizes[0] =
                rv64m.range_tuple_checker_sizes[0].max(bigint.range_tuple_checker_sizes[0]);
            rv64m.range_tuple_checker_sizes[1] =
                rv64m.range_tuple_checker_sizes[1].max(bigint.range_tuple_checker_sizes[1]);
            bigint.range_tuple_checker_sizes = rv64m.range_tuple_checker_sizes;
        }

        const DEFERRAL_AS_USIZE: usize = DEFERRAL_AS as usize;
        let addr_spaces = &mut self.system.config.memory_config.addr_spaces;
        let deferral_as_exists = addr_spaces.len() > DEFERRAL_AS_USIZE;
        if self.deferral.is_some() {
            assert!(
                deferral_as_exists,
                "deferral is enabled but address space DEFERRAL_AS ({DEFERRAL_AS_USIZE}) is missing \
                 from memory_config.addr_spaces; the VM config must allocate it"
            );
        } else if deferral_as_exists {
            addr_spaces[DEFERRAL_AS_USIZE].num_cells = 0;
        }
    }

    pub fn to_inner(&self) -> SdkVmConfigInner {
        let config = self.clone().optimize();
        let system = config.system.config.clone();
        let rv64i = config.rv64i.map(|_| Rv64I);
        let io = config.io.map(|_| Rv64Io);
        let keccak = config.keccak.map(|_| Keccak256);
        let sha2 = config.sha2.map(|_| Sha2);
        let rv64m = config.rv64m;
        let bigint = config.bigint;
        let modular = config.modular.clone();
        let fp2 = config.fp2.clone();
        let pairing = config.pairing.clone();
        let ecc = config.ecc.clone();
        let deferral = config.deferral.as_ref().map(DeferralConfig::to_extension);

        SdkVmConfigInner {
            system,
            rv64i,
            io,
            keccak,
            sha2,
            rv64m,
            bigint,
            modular,
            fp2,
            pairing,
            ecc,
            deferral,
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
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension(executor = "Rv64IExecutor")]
    pub rv64i: Option<Rv64I>,
    #[extension(executor = "Rv64IoExecutor")]
    pub io: Option<Rv64Io>,
    #[extension(executor = "Keccak256Executor")]
    pub keccak: Option<Keccak256>,
    #[extension(executor = "Sha2Executor")]
    pub sha2: Option<Sha2>,

    #[extension(executor = "Rv64MExecutor")]
    pub rv64m: Option<Rv64M>,
    #[extension(executor = "Int256Executor")]
    pub bigint: Option<Int256>,
    #[extension(executor = "ModularExtensionExecutor")]
    pub modular: Option<ModularExtension>,
    #[extension(executor = "Fp2ExtensionExecutor")]
    pub fp2: Option<Fp2Extension>,
    #[extension(executor = "PairingExtensionExecutor")]
    pub pairing: Option<PairingExtension>,
    #[extension(executor = "WeierstrassExtensionExecutor")]
    pub ecc: Option<WeierstrassExtension>,

    #[extension(executor = "DeferralExecutor")]
    pub deferral: Option<DeferralExtension>,
}

// Generated by macro
pub type SdkVmConfigExecutor = SdkVmConfigInnerExecutor;

#[cfg(feature = "rvr")]
impl<F, RA> VmRvrLogNativeExtension<F, RA> for SdkVmConfigInner
where
    F: PrimeField32,
    RA: Arena
        + Rv64IRecordArena<F>
        + Rv64MRecordArena<F>
        + Rv64IoRecordArena<F>
        + ModularRecordArena<F>
        + WeierstrassRecordArena<F>
        + KeccakRecordArena<F>
        + Sha256RecordArena<F>
        + Int256RecordArena<F>,
{
    fn extend_rvr_log_native(&self, registry: &mut LogNativeAssemblerRegistry<F, RA>) {
        // Inner-to-outer ownership order: RV64I owns PHANTOM, then RV64M/IO,
        // modular dependencies, and finally each independent extension.
        self.rv64i.extend_rvr_log_native(registry);
        self.rv64m.extend_rvr_log_native(registry);
        self.io.extend_rvr_log_native(registry);
        self.modular.extend_rvr_log_native(registry);
        self.fp2.extend_rvr_log_native(registry);
        self.ecc.extend_rvr_log_native(registry);
        self.keccak.extend_rvr_log_native(registry);
        self.sha2.extend_rvr_log_native(registry);
        self.bigint.extend_rvr_log_native(registry);
        self.pairing.extend_rvr_log_native(registry);
    }
}

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

    #[cfg(feature = "rvr")]
    fn create_rvr_extensions(&self, air_idx: Option<&[usize]>) -> RvrExtensions
    where
        F: PrimeField32,
    {
        self.to_inner().create_rvr_extensions(air_idx)
    }
}

impl<SC: StarkProtocolConfig> VmCircuitConfig<SC> for SdkVmConfig
where
    SdkVmConfigInner: VmCircuitConfig<SC>,
{
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError> {
        self.to_inner().create_airs()
    }
}

use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
type SC = BabyBearPoseidon2Config;
impl<E> VmBuilder<E> for SdkVmCpuBuilder
where
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
{
    type VmConfig = SdkVmConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &SdkVmConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let config = config.to_inner();
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        if let Some(rv64i) = &config.rv64i {
            VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, rv64i, inventory)?;
        }
        if let Some(io) = &config.io {
            VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, io, inventory)?;
        }
        if let Some(keccak) = &config.keccak {
            VmProverExtension::<E, _, _>::extend_prover(&Keccak256CpuProverExt, keccak, inventory)?;
        }
        if let Some(sha2) = &config.sha2 {
            VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, sha2, inventory)?;
        }
        if let Some(rv64m) = &config.rv64m {
            VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, rv64m, inventory)?;
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
        if let Some(deferral) = &config.deferral {
            VmProverExtension::<E, _, _>::extend_prover(
                &DeferralCpuProverExt,
                deferral,
                inventory,
            )?;
        }
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<E::SC>, Self::RecordArena>
    where
        Val<E::SC>: PrimeField32,
    {
        let config = config.to_inner();
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Default)]
pub struct SdkVmGpuBuilder {
    /// M-GPUDEC shared decode state: one per VM, cloned into the rv64im GPU
    /// prover extension so a bound compact operand table reaches the chips.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<openvm_riscv_circuit::rvr_gpu_decode::RvrGpuDecodeState>,
}

#[cfg(feature = "cuda")]
impl VmBuilder<BabyBearPoseidon2GpuEngine> for SdkVmGpuBuilder {
    type VmConfig = SdkVmConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &SdkVmConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<BabyBearPoseidon2GpuEngine>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        type E = BabyBearPoseidon2GpuEngine;

        let config = config.to_inner();
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        #[cfg(feature = "rvr")]
        chip_complex
            .system
            .set_device_touched_memory_provider(self.rvr_decode.clone());
        let inventory = &mut chip_complex.inventory;
        let rv64im_ext = Rv64ImGpuProverExt {
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            rvr_decode: self.rvr_decode.clone(),
        };
        if let Some(rv64i) = &config.rv64i {
            VmProverExtension::<E, _, _>::extend_prover(&rv64im_ext, rv64i, inventory)?;
        }
        if let Some(io) = &config.io {
            VmProverExtension::<E, _, _>::extend_prover(&rv64im_ext, io, inventory)?;
        }
        if let Some(keccak) = &config.keccak {
            VmProverExtension::<E, _, _>::extend_prover(&Keccak256GpuProverExt, keccak, inventory)?;
        }
        if let Some(sha2) = &config.sha2 {
            VmProverExtension::<E, _, _>::extend_prover(&Sha2GpuProverExt, sha2, inventory)?;
        }
        if let Some(rv64m) = &config.rv64m {
            VmProverExtension::<E, _, _>::extend_prover(&rv64im_ext, rv64m, inventory)?;
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
        if let Some(deferral) = &config.deferral {
            VmProverExtension::<E, _, _>::extend_prover(&DeferralProverExt, deferral, inventory)?;
        }
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<SC>, Self::RecordArena>
    where
        Val<SC>: PrimeField32,
    {
        let config = config.to_inner();
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }

    /// M-GPUDEC: same compact wire-arena adoption as the rv64im GPU builders,
    /// against this builder's shared decode state (the chips hold clones of
    /// it via `create_chip_complex`). Airs outside the migrated rv64im set
    /// simply keep their expanded arenas.
    #[cfg(feature = "rvr")]
    fn generate_rvr_record_arenas_from_logs(
        &self,
        config: &Self::VmConfig,
        exe: &openvm_circuit::arch::instructions::exe::VmExe<Val<SC>>,
        output: &mut openvm_circuit::arch::rvr::RvrPreflightOutput<Val<SC>>,
        capacities: &[(usize, usize)],
        pc_to_air_idx: &[Option<usize>],
    ) -> Result<Option<Vec<Self::RecordArena>>, openvm_circuit::arch::ExecutionError>
    where
        Val<SC>: PrimeField32,
    {
        let registry = self.create_rvr_log_native_assembler_registry(config);
        openvm_riscv_circuit::generate_gpu_rvr_record_arenas(
            &registry,
            &self.rvr_decode,
            exe,
            output,
            capacities,
            pc_to_air_idx,
        )
    }

    /// G2: same wire-staging opt-in as the rv64im GPU builders, against this
    /// builder's shared decode state (a composed builder that adopts compact
    /// arenas without staging would silently fall back to the alloc+memcpy
    /// path — the gate-#4 threading lesson).
    #[cfg(feature = "rvr")]
    fn rvr_wire_record_airs(
        &self,
        _config: &Self::VmConfig,
        exe: &openvm_circuit::arch::instructions::exe::VmExe<Val<SC>>,
        pc_to_air_idx: &[Option<usize>],
        inline_meta: &openvm_circuit::arch::rvr::RvrInlineRecordsMeta,
    ) -> std::collections::HashSet<usize>
    where
        Val<SC>: PrimeField32,
    {
        openvm_riscv_circuit::rvr_gpu_wire_record_airs(
            &self.rvr_decode,
            exe,
            pc_to_air_idx,
            inline_meta,
        )
    }

    /// GPU backend: default to the rvr inline preflight engine — the host
    /// compact→arena assembly pass that dominates the CPU path does not
    /// exist in the GPU shape, and compact records shrink the H2D payload.
    #[cfg(feature = "rvr")]
    fn default_rvr_preflight_engine(&self) -> openvm_circuit::arch::rvr::RvrPreflightEngine {
        openvm_circuit::arch::rvr::RvrPreflightEngine::Rvr
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

impl From<Rv64I> for UnitStruct {
    fn from(_: Rv64I) -> Self {
        UnitStruct {}
    }
}

impl From<Rv64Io> for UnitStruct {
    fn from(_: Rv64Io) -> Self {
        UnitStruct {}
    }
}

impl From<Keccak256> for UnitStruct {
    fn from(_: Keccak256) -> Self {
        UnitStruct {}
    }
}

impl From<Sha2> for UnitStruct {
    fn from(_: Sha2) -> Self {
        UnitStruct {}
    }
}

#[derive(Deserialize)]
struct SdkVmConfigWithDefaultDeser {
    #[serde(default)]
    pub system: SdkSystemConfig,

    pub rv64i: Option<UnitStruct>,
    pub io: Option<UnitStruct>,
    pub keccak: Option<UnitStruct>,
    pub sha2: Option<UnitStruct>,

    pub rv64m: Option<Rv64M>,
    pub bigint: Option<Int256>,
    pub modular: Option<ModularExtension>,
    pub fp2: Option<Fp2Extension>,
    pub pairing: Option<PairingExtension>,
    pub ecc: Option<WeierstrassExtension>,

    pub deferral: Option<DeferralConfig>,
}

impl From<SdkVmConfigWithDefaultDeser> for SdkVmConfig {
    fn from(config: SdkVmConfigWithDefaultDeser) -> Self {
        let ret = Self {
            system: config.system,
            rv64i: config.rv64i,
            io: config.io,
            keccak: config.keccak,
            sha2: config.sha2,
            rv64m: config.rv64m,
            bigint: config.bigint,
            modular: config.modular,
            fp2: config.fp2,
            pairing: config.pairing,
            ecc: config.ecc,
            deferral: config.deferral,
        };
        ret.optimize()
    }
}

#[cfg(all(test, feature = "rvr"))]
mod rvr_registration_tests {
    use openvm_algebra_transpiler::{Fp2Opcode, Rv64ModularArithmeticOpcode};
    use openvm_instructions::{
        instruction::Instruction,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        LocalOpcode, SystemOpcode, VmOpcode,
    };

    use super::*;

    fn extension_instruction(opcode: VmOpcode) -> Instruction<F> {
        Instruction::from_usize(
            opcode,
            [1, 2, 3, RV64_REGISTER_AS as usize, RV64_MEMORY_AS as usize],
        )
    }

    #[test]
    fn standard_rvr_registry_admits_all_composed_extension_opcodes() {
        let config = SdkVmConfig::standard().to_inner();
        let mut registry = LogNativeAssemblerRegistry::<F, MatrixRecordArena<F>>::new();
        config.extend_rvr_log_native(&mut registry);

        // HARD-1: the composed registry has one shared PHANTOM owner (RV64I),
        // including for pairing's final-exp hint.
        assert!(registry.contains_instruction(&Instruction::from_usize(
            SystemOpcode::PHANTOM.global_opcode(),
            [0; 5],
        )));

        for opcode in [
            KeccakfOpcode::KECCAKF.global_opcode(),
            XorinOpcode::XORIN.global_opcode(),
        ] {
            assert!(registry.contains_instruction(&extension_instruction(opcode)));
        }

        assert!(registry.contains_instruction(&extension_instruction(
            Rv64Sha2Opcode::SHA256.global_opcode(),
        )));
        assert!(
            !registry.contains_instruction(&extension_instruction(
                Rv64Sha2Opcode::SHA512.global_opcode(),
            )),
            "SHA-512 has no log-native assembler and must not be silently admitted as SHA-256"
        );

        let modular = config.modular.as_ref().expect("standard modular config");
        let modular_opcodes = [
            Rv64ModularArithmeticOpcode::ADD,
            Rv64ModularArithmeticOpcode::SUB,
            Rv64ModularArithmeticOpcode::SETUP_ADDSUB,
            Rv64ModularArithmeticOpcode::MUL,
            Rv64ModularArithmeticOpcode::DIV,
            Rv64ModularArithmeticOpcode::SETUP_MULDIV,
            Rv64ModularArithmeticOpcode::IS_EQ,
            Rv64ModularArithmeticOpcode::SETUP_ISEQ,
        ];
        for modulus_idx in 0..modular.supported_moduli.len() {
            for local_opcode in modular_opcodes {
                let opcode = VmOpcode::from_usize(
                    Rv64ModularArithmeticOpcode::CLASS_OFFSET
                        + modulus_idx * modular_opcodes.len()
                        + local_opcode as usize,
                );
                assert!(registry.contains_instruction(&extension_instruction(opcode)));
            }
        }

        let fp2 = config.fp2.as_ref().expect("standard Fp2 config");
        let fp2_opcodes = [
            Fp2Opcode::ADD,
            Fp2Opcode::SUB,
            Fp2Opcode::SETUP_ADDSUB,
            Fp2Opcode::MUL,
            Fp2Opcode::DIV,
            Fp2Opcode::SETUP_MULDIV,
        ];
        for modulus_idx in 0..fp2.supported_moduli.len() {
            for local_opcode in fp2_opcodes {
                let opcode = VmOpcode::from_usize(
                    Fp2Opcode::CLASS_OFFSET
                        + modulus_idx * fp2_opcodes.len()
                        + local_opcode as usize,
                );
                assert!(registry.contains_instruction(&extension_instruction(opcode)));
            }
        }

        let ecc = config.ecc.as_ref().expect("standard ECC config");
        let ecc_opcodes = [
            Rv64WeierstrassOpcode::EC_ADD_NE,
            Rv64WeierstrassOpcode::SETUP_EC_ADD_NE,
            Rv64WeierstrassOpcode::EC_DOUBLE,
            Rv64WeierstrassOpcode::SETUP_EC_DOUBLE,
        ];
        for curve_idx in 0..ecc.supported_curves.len() {
            for local_opcode in ecc_opcodes {
                let opcode = VmOpcode::from_usize(
                    Rv64WeierstrassOpcode::CLASS_OFFSET
                        + curve_idx * ecc_opcodes.len()
                        + local_opcode as usize,
                );
                assert!(registry.contains_instruction(&extension_instruction(opcode)));
            }
        }

        for opcode in Rv64BaseAlu256Opcode::iter()
            .map(|opcode| opcode.global_opcode())
            .chain(Rv64Shift256Opcode::iter().map(|opcode| opcode.global_opcode()))
            .chain(Rv64LessThan256Opcode::iter().map(|opcode| opcode.global_opcode()))
            .chain(Rv64BranchEqual256Opcode::iter().map(|opcode| opcode.global_opcode()))
            .chain(Rv64BranchLessThan256Opcode::iter().map(|opcode| opcode.global_opcode()))
            .chain(Rv64Mul256Opcode::iter().map(|opcode| opcode.global_opcode()))
        {
            assert!(registry.contains_instruction(&extension_instruction(opcode)));
        }

        let wrong_address_spaces = Instruction::from_usize(
            Rv64ModularArithmeticOpcode::ADD.global_opcode(),
            [1, 2, 3, 0, 0],
        );
        assert!(
            !registry.contains_instruction(&wrong_address_spaces),
            "HARD-2 predicate must reject the same operands for routing and assembly"
        );
    }
}
