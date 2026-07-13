#![cfg_attr(feature = "tco", allow(incomplete_features))]
#![cfg_attr(feature = "tco", feature(explicit_tail_calls))]
#![cfg_attr(feature = "tco", allow(internal_features))]
#![cfg_attr(feature = "tco", feature(core_intrinsics))]
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::{LogNativeAssemblerRegistry, VmRvrLogNativeExtension};
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventoryError, InitFileGenerator, MatrixRecordArena, SystemConfig,
        VmBuilder, VmChipComplex, VmField, VmProverExtension,
    },
    system::{SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{Executor, MeteredExecutor, PreflightExecutor, VmConfig};
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

#[cfg(feature = "rvr")]
pub mod log_native;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::{extensions::SystemGpuBuilder, SystemChipInventoryGPU};
        use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        pub(crate) mod cuda_abi;
        pub use self::{
            Rv64IGpuBuilder as Rv64IBuilder,
            Rv64ImGpuBuilder as Rv64ImBuilder,
        };
    } else {
        pub use self::{
            Rv64ICpuBuilder as Rv64IBuilder,
            Rv64ImCpuBuilder as Rv64ImBuilder,
        };
    }
}

#[cfg(all(test, feature = "rvr"))]
mod rvr_preflight_tests;
#[cfg(any(test, feature = "test-utils"))]
mod test_utils;

// Config for a VM with base extension and IO extension
#[derive(Clone, Debug, derive_new::new, VmConfig, Serialize, Deserialize)]
pub struct Rv64IConfig {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv64I,
    #[extension]
    pub io: Rv64Io,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv64IConfig {}

/// Config for a VM with base extension, IO extension, and multiplication extension
#[derive(Clone, Debug, Default, VmConfig, derive_new::new, Serialize, Deserialize)]
pub struct Rv64ImConfig {
    #[config]
    pub rv64i: Rv64IConfig,
    #[extension]
    pub mul: Rv64M,
}

// Default implementation uses no init file
impl InitFileGenerator for Rv64ImConfig {}

impl Default for Rv64IConfig {
    fn default() -> Self {
        let system = SystemConfig::default();
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv64IConfig {
    pub fn with_public_values_bytes(num_public_values_bytes: usize) -> Self {
        let system = SystemConfig::default().with_public_values_bytes(num_public_values_bytes);
        Self {
            system,
            base: Default::default(),
            io: Default::default(),
        }
    }
}

impl Rv64ImConfig {
    pub fn with_public_values_bytes(num_public_values_bytes: usize) -> Self {
        Self {
            rv64i: Rv64IConfig::with_public_values_bytes(num_public_values_bytes),
            mul: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct Rv64ICpuBuilder;

impl<SC, E> VmBuilder<E> for Rv64ICpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv64IConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Rv64IConfig,
        circuit: AirInventory<E::SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.io, inventory)?;
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<E::SC>, Self::RecordArena>
    where
        Val<E::SC>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }
}

#[derive(Clone)]
pub struct Rv64ImCpuBuilder;

impl<SC, E> VmBuilder<E> for Rv64ImCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv64ImConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<E::SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ICpuBuilder,
            &config.rv64i,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.mul, inventory)?;
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<E::SC>, Self::RecordArena>
    where
        Val<E::SC>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }
}

// The operand-table classifier is host-only and is also the independent CPU
// oracle for compact compiler metadata; only its device upload is CUDA-gated.
#[cfg(feature = "rvr")]
pub mod rvr_gpu_decode;

#[cfg(feature = "cuda")]
#[derive(Clone, Default)]
pub struct Rv64IGpuBuilder {
    /// M-GPUDEC shared producer/consumer state (device operand table +
    /// per-segment emission modes); cloned into migrated GPU chips.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<rvr_gpu_decode::RvrGpuDecodeState>,
}

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for Rv64IGpuBuilder {
    type VmConfig = Rv64IConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64IConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &EngineDeviceCtx<GpuBabyBearPoseidon2Engine>,
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
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        let prover_ext = Rv64ImGpuProverExt {
            #[cfg(all(feature = "cuda", feature = "rvr"))]
            rvr_decode: self.rvr_decode.clone(),
        };
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &prover_ext,
            &config.base,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &prover_ext,
            &config.io,
            inventory,
        )?;
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<BabyBearPoseidon2Config>, Self::RecordArena>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }

    #[cfg(feature = "rvr")]
    fn generate_rvr_record_arenas_from_logs(
        &self,
        config: &Self::VmConfig,
        exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
        output: &mut openvm_circuit::arch::rvr::RvrPreflightOutput<Val<BabyBearPoseidon2Config>>,
        capacities: &[(usize, usize)],
        pc_to_air_idx: &[Option<usize>],
    ) -> Result<Option<Vec<Self::RecordArena>>, openvm_circuit::arch::ExecutionError>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let registry = self.create_rvr_log_native_assembler_registry(config);
        generate_gpu_rvr_record_arenas(
            &registry,
            &self.rvr_decode,
            exe,
            output,
            capacities,
            pc_to_air_idx,
        )
    }

    #[cfg(feature = "rvr")]
    fn rvr_wire_record_airs(
        &self,
        _config: &Self::VmConfig,
        exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
        pc_to_air_idx: &[Option<usize>],
    ) -> std::collections::HashSet<usize>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        rvr_gpu_wire_record_airs(&self.rvr_decode, exe, pc_to_air_idx)
    }

    /// GPU backend: default to the rvr inline preflight engine — the host
    /// compact→arena assembly pass that dominates the CPU path does not
    /// exist in the GPU shape, and compact records shrink the H2D payload.
    #[cfg(feature = "rvr")]
    fn default_rvr_preflight_engine(&self) -> openvm_circuit::arch::rvr::RvrPreflightEngine {
        openvm_circuit::arch::rvr::RvrPreflightEngine::Rvr
    }
}

/// G2: the airs whose inline records the proving path should stage as compact
/// wire targets for this shared decode `state` — the device-decodable set,
/// bound per exe (taint keeps mixed programs correct). Empty unless
/// `OPENVM_RVR_GPU_RECORDS=compact`. Shared by every wired GPU builder's
/// `rvr_wire_record_airs` override, including composed-config builders.
#[cfg(all(feature = "cuda", feature = "rvr"))]
pub fn rvr_gpu_wire_record_airs(
    _state: &rvr_gpu_decode::RvrGpuDecodeState,
    exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
    pc_to_air_idx: &[Option<usize>],
) -> std::collections::HashSet<usize> {
    use crate::rvr_gpu_decode::{configured_emission_mode, InlineEmissionMode};
    match configured_emission_mode() {
        Some(InlineEmissionMode::CompactWire) => {
            rvr_gpu_decode::RvrGpuDecodeState::compact_record_airs(exe, pc_to_air_idx)
        }
        // Delta owns one global backing, not per-AIR host wire targets.
        Some(InlineEmissionMode::Delta) => Default::default(),
        _ => Default::default(),
    }
}

/// Shared M-GPUDEC record-arena hook for the GPU builders: with
/// `OPENVM_RVR_GPU_RECORDS=compact`, migrated AIRs' records stay in wire form
/// for on-device decode; default keeps the gate-validated expanded path.
/// Public so composed-config GPU builders (e.g. the SDK builder) can opt their
/// VMs into the same compact path with their own shared decode state.
///
/// On the proving path the wire records already sit in C-staged arena
/// backings (zero-copy — see [`rvr_gpu_wire_record_airs`]) and arrive here
/// only as counts to verify; the adoption loop below is the UNSTAGED fallback
/// (one alloc + memcpy per compact air), used by harnesses that execute
/// without arena targets.
#[cfg(all(feature = "cuda", feature = "rvr"))]
#[allow(clippy::type_complexity)]
pub fn generate_gpu_rvr_record_arenas(
    registry: &LogNativeAssemblerRegistry<Val<BabyBearPoseidon2Config>, DenseRecordArena>,
    state: &rvr_gpu_decode::RvrGpuDecodeState,
    exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
    output: &mut openvm_circuit::arch::rvr::RvrPreflightOutput<Val<BabyBearPoseidon2Config>>,
    capacities: &[(usize, usize)],
    pc_to_air_idx: &[Option<usize>],
) -> Result<Option<Vec<DenseRecordArena>>, openvm_circuit::arch::ExecutionError> {
    use openvm_circuit::arch::rvr::generate_record_arenas_from_logs_with_compact;

    use crate::rvr_gpu_decode::{configured_emission_mode, InlineEmissionMode};
    if registry.is_empty() {
        return Ok(None);
    }
    let compact_requested = matches!(
        configured_emission_mode(),
        Some(InlineEmissionMode::CompactWire)
    );
    let delta_requested = matches!(configured_emission_mode(), Some(InlineEmissionMode::Delta));
    if !delta_requested {
        state.clear_delta_segment();
    }
    let compact_airs = if compact_requested {
        state.bind_compact_segment(exe, pc_to_air_idx, &output.inline_pc_slots)
    } else if delta_requested {
        state.bind_delta_airs(exe, pc_to_air_idx, &output.inline_pc_slots)
    } else {
        Default::default()
    };
    let saved_delta = if delta_requested {
        Some(output.delta_records.take().ok_or_else(|| {
            openvm_circuit::arch::ExecutionError::RvrExecution(
                "GPU delta mode requested but native preflight emitted no delta backing"
                    .to_string(),
            )
        })?)
    } else {
        None
    };
    let (mut arenas, wire_buffers) = generate_record_arenas_from_logs_with_compact(
        registry,
        exe,
        output,
        capacities,
        pc_to_air_idx,
        &compact_airs,
    )?;
    if delta_requested && !wire_buffers.is_empty() {
        return Err(openvm_circuit::arch::ExecutionError::RvrExecution(
            "GPU delta mode unexpectedly produced host compact buffers".to_string(),
        ));
    }
    // Silent-mode guard (the gate-#4 class): under the compact opt-in every
    // decodable air must be accounted for — either C-staged by the caller
    // (reported in arena_native_written) or adopted below. An air that is
    // neither ran a different emission than the label claims.
    if compact_requested {
        for &air in &compact_airs {
            let staged = output
                .arena_native_written
                .iter()
                .any(|&(written_air, _)| written_air == air);
            let adopted = wire_buffers.iter().any(|chip| chip.air_idx == air);
            if !staged && !adopted {
                return Err(openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                    "compact air {air} was neither wire-staged nor adopted; the compiled \
                     library did not emit compact wire records for it (fused-compiled \
                     library under a compact opt-in?)"
                )));
            }
        }
    }
    for chip in wire_buffers {
        let arena = arenas.get_mut(chip.air_idx).ok_or_else(|| {
            openvm_circuit::arch::ExecutionError::RvrExecution(format!(
                "compact air_idx {} out of arena range",
                chip.air_idx
            ))
        })?;
        // Unstaged fallback adoption: one alloc + memcpy of the wire buffer
        // (NOT zero-copy — the zero-copy path is the staged one above).
        let mut dense = DenseRecordArena::with_byte_capacity(chip.bytes.len());
        dense
            .alloc_bytes(chip.bytes.len())
            .copy_from_slice(&chip.bytes);
        // The mode travels with the data: this is what routes the arena to the
        // chip's compact-decode branch instead of the expanded-record kernel.
        dense.rvr_wire = true;
        *arena = dense;
    }
    if let Some(delta) = saved_delta {
        let memory_log = std::mem::take(&mut output.raw_logs.memory_log);
        let program_log = std::mem::take(&mut output.raw_logs.program_log);
        let chip_counts = std::mem::take(&mut output.raw_logs.chip_counts);
        let bound_airs = state.bind_delta_segment(
            exe,
            pc_to_air_idx,
            &output.inline_pc_slots,
            delta,
            memory_log,
            program_log,
            chip_counts,
            &output.arena_native_written,
        )?;
        if !bound_airs.is_subset(&compact_airs) {
            return Err(openvm_circuit::arch::ExecutionError::RvrExecution(
                "GPU delta bound AIR set drifted from the compiled decode set".to_string(),
            ));
        }
    }
    Ok(Some(arenas))
}

#[cfg(feature = "cuda")]
#[derive(Clone, Default)]
pub struct Rv64ImGpuBuilder {
    /// See [`Rv64IGpuBuilder::rvr_decode`]; one shared state per VM.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    pub rvr_decode: std::sync::Arc<rvr_gpu_decode::RvrGpuDecodeState>,
}

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for Rv64ImGpuBuilder {
    type VmConfig = Rv64ImConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device_ctx: &EngineDeviceCtx<GpuBabyBearPoseidon2Engine>,
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
            &Rv64IGpuBuilder {
                #[cfg(all(feature = "cuda", feature = "rvr"))]
                rvr_decode: self.rvr_decode.clone(),
            },
            &config.rv64i,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv64ImGpuProverExt {
                #[cfg(all(feature = "cuda", feature = "rvr"))]
                rvr_decode: self.rvr_decode.clone(),
            },
            &config.mul,
            inventory,
        )?;
        Ok(chip_complex)
    }

    #[cfg(feature = "rvr")]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<BabyBearPoseidon2Config>, Self::RecordArena>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let mut registry = LogNativeAssemblerRegistry::new();
        config.extend_rvr_log_native(&mut registry);
        registry
    }

    #[cfg(feature = "rvr")]
    fn generate_rvr_record_arenas_from_logs(
        &self,
        config: &Self::VmConfig,
        exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
        output: &mut openvm_circuit::arch::rvr::RvrPreflightOutput<Val<BabyBearPoseidon2Config>>,
        capacities: &[(usize, usize)],
        pc_to_air_idx: &[Option<usize>],
    ) -> Result<Option<Vec<Self::RecordArena>>, openvm_circuit::arch::ExecutionError>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        let registry = self.create_rvr_log_native_assembler_registry(config);
        generate_gpu_rvr_record_arenas(
            &registry,
            &self.rvr_decode,
            exe,
            output,
            capacities,
            pc_to_air_idx,
        )
    }

    #[cfg(feature = "rvr")]
    fn rvr_wire_record_airs(
        &self,
        _config: &Self::VmConfig,
        exe: &openvm_instructions::exe::VmExe<Val<BabyBearPoseidon2Config>>,
        pc_to_air_idx: &[Option<usize>],
    ) -> std::collections::HashSet<usize>
    where
        Val<BabyBearPoseidon2Config>: openvm_stark_backend::p3_field::PrimeField32,
    {
        rvr_gpu_wire_record_airs(&self.rvr_decode, exe, pc_to_air_idx)
    }

    /// GPU backend: default to the rvr inline preflight engine — the host
    /// compact→arena assembly pass that dominates the CPU path does not
    /// exist in the GPU shape, and compact records shrink the H2D payload.
    #[cfg(feature = "rvr")]
    fn default_rvr_preflight_engine(&self) -> openvm_circuit::arch::rvr::RvrPreflightEngine {
        openvm_circuit::arch::rvr::RvrPreflightEngine::Rvr
    }
}
