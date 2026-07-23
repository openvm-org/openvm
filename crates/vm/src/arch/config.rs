use std::{
    fs::File,
    io::{self, Write},
    mem::size_of,
    path::Path,
};

use derive_new::new;
use getset::{Setters, WithSetters};
#[cfg(feature = "rvr")]
use openvm_instructions::exe::VmExe;
use openvm_instructions::{
    riscv::{RV64_IMM_AS, RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS, PUBLIC_VALUES_AS, VM_DIGEST_WIDTH,
};
use openvm_platform::memory::MEM_SIZE;
use openvm_poseidon2_air::Poseidon2Config;
#[cfg(feature = "rvr")]
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::{
    p3_field::Field, EngineDeviceCtx, StarkEngine, StarkProtocolConfig, Val,
};
#[cfg(feature = "rvr")]
use rvr_openvm_lift::RvrExtensions;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[cfg(feature = "rvr")]
use super::ExecutionError;
use super::{AnyEnum, VmChipComplex, BOUNDARY_AIR_ID, CONNECTOR_AIR_ID, PROGRAM_AIR_ID};
#[cfg(feature = "rvr")]
use crate::arch::rvr::RvrPreflightOutput;
#[cfg(feature = "rvr")]
use crate::arch::rvr::{generate_record_arenas_from_logs, LogNativeAssemblerRegistry};
use crate::{
    arch::{
        execution_mode::metered::segment_ctx::DEFAULT_MAX_MEMORY, AirInventory, AirInventoryError,
        Arena, ChipInventoryError, ExecutorInventory, ExecutorInventoryError,
    },
    system::{
        memory::{
            merkle::public_values::{assert_public_values_shape, public_values_cells_from_bytes},
            num_memory_airs, POINTER_MAX_BITS,
        },
        SystemChipComplex,
    },
};

// sbox is decomposed to have this max degree for Poseidon2. We set to 3 so quotient_degree = 2
// allows log_blowup = 1
const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 3;
pub const DEFAULT_MAX_NUM_PUBLIC_VALUES: usize = 32;
/// Max number of deferral address space cells
pub const DEFAULT_DEFERRAL_ADDR_SPACE_CELLS: usize = 1 << 14;
/// Width of Poseidon2 VM uses.
pub const POSEIDON2_WIDTH: usize = 2 * VM_DIGEST_WIDTH;
/// Offset for address space indices. This is used to distinguish between different memory spaces.
pub const ADDR_SPACE_OFFSET: u32 = 1;

fn default_segmentation_max_memory() -> usize {
    DEFAULT_MAX_MEMORY
}

pub const OPENVM_DEFAULT_INIT_FILE_BASENAME: &str = "openvm_init";
pub const OPENVM_DEFAULT_INIT_FILE_NAME: &str = "openvm_init.rs";

// Memory-layout constants. Mirror the CUDA-side constants in
// `crates/vm/cuda/include/system/memory/params.cuh` (which also contains the
// byte/cell/block/leaf layout diagram).
//
// Terminology:
//   Cell    one storage word in an address space.
//   Block   the unit of one memory-bus message: BLOCK_FE_WIDTH cells =
//           MEMORY_BLOCK_BYTES bytes.
//   Digest  the output of one Poseidon2 compression (VM_DIGEST_WIDTH cells); also
//           one merkle leaf.

/// Host byte width of one u16-celled storage cell.
pub const U16_CELL_SIZE: usize = size_of::<u16>();

// TODO: replace with `p3_util::log2_strict_usize` once p3-util is bumped to
// >= 0.4.3 (where it becomes `const fn`).
pub(crate) const fn const_log2_strict_usize(value: usize) -> usize {
    assert!(value.is_power_of_two(), "value must be a power of two");
    value.ilog2() as usize
}

/// log2 of [`U16_CELL_SIZE`].
pub const U16_CELL_SIZE_BITS: usize = const_log2_strict_usize(U16_CELL_SIZE);

/// Converts pointer bits for a u16-celled address space to byte-pointer bits.
pub const fn to_byte_ptr_bits(ptr_bits: usize) -> usize {
    ptr_bits + U16_CELL_SIZE_BITS
}

/// Cells per memory-bus block.
pub const BLOCK_FE_WIDTH: usize = 4;

/// Bytes per memory-bus block.
pub const MEMORY_BLOCK_BYTES: usize = BLOCK_FE_WIDTH * U16_CELL_SIZE;

// TODO: make executor debug bounds use `MemoryConfig::pointer_max_bits` once
// execution state carries the memory config.

/// Number of registers in the RV64 register file.
pub const NUM_RV64_REGISTERS: usize = 32;

/// Returns a Poseidon2 config for the VM.
pub fn vm_poseidon2_config<F: Field>() -> Poseidon2Config<F> {
    Poseidon2Config::default()
}

/// A VM configuration is the minimum serializable format to be able to create the execution
/// environment and circuit for a zkVM supporting a fixed set of instructions.
/// This trait contains the sub-traits [VmExecutionConfig] and [VmCircuitConfig].
/// The [InitFileGenerator] sub-trait provides custom build hooks to generate code for initializing
/// some VM extensions. The `VmConfig` is expected to contain the [SystemConfig] internally.
///
/// For users who only need to create an execution environment, use the sub-trait
/// [VmExecutionConfig] to avoid the `SC` generic.
///
/// This trait does not contain the [VmBuilder] trait, because a single VM configuration may
/// implement multiple [VmBuilder]s for different prover backends.
pub trait VmConfig<SC>:
    Clone
    + Serialize
    + DeserializeOwned
    + InitFileGenerator
    + VmExecutionConfig<Val<SC>>
    + VmCircuitConfig<SC>
    + AsRef<SystemConfig>
    + AsMut<SystemConfig>
where
    SC: StarkProtocolConfig,
{
}

pub trait VmExecutionConfig<F> {
    type Executor: AnyEnum;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;

    #[cfg(feature = "rvr")]
    fn create_rvr_extensions(&self, air_idx: Option<&[usize]>) -> RvrExtensions
    where
        F: PrimeField32;
}

pub trait VmCircuitConfig<SC: StarkProtocolConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}

/// This trait is intended to be implemented on a new type wrapper of the VmConfig struct to get
/// around Rust orphan rules.
pub trait VmBuilder<E: StarkEngine>: Sized {
    type VmConfig: VmConfig<E::SC>;
    /// With the rvr feature, arenas must also know how to stage themselves
    /// as R4 arena-native write targets for the generated C.
    #[cfg(feature = "rvr")]
    type RecordArena: Arena + Send + crate::arch::rvr::preflight::RvrArenaNativeTarget;
    #[cfg(not(feature = "rvr"))]
    type RecordArena: Arena;
    type SystemChipInventory: SystemChipComplex<Self::RecordArena, E::PB>;

    /// Create a [VmChipComplex] from the full [AirInventory], which should be the output of
    /// [VmCircuitConfig::create_airs].
    #[allow(clippy::type_complexity)]
    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<E::SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    >;

    /// Default preflight engine for the proving path when neither the
    /// per-instance override nor `OPENVM_RVR_PREFLIGHT_ENGINE` is set.
    ///
    /// CPU prover builders keep the trait default (`Interpreter`): at reth
    /// scale the interpreter's fused execute+arena-fill pass beats the rvr
    /// inline path, whose host compact→arena assembly dominates its cost on
    /// CPU (see [`crate::arch::rvr::RvrPreflightEngine`] for the measured
    /// rationale). GPU builders override to `Rvr`: the assembly pass does not
    /// exist in the GPU shape, and compact records shrink the H2D payload.
    #[cfg(feature = "rvr")]
    fn default_rvr_preflight_engine(&self) -> crate::arch::rvr::RvrPreflightEngine {
        crate::arch::rvr::RvrPreflightEngine::Interpreter
    }

    /// CUDA/G2: create an owned module/context prewarm task. GPU builders
    /// override this so it can run concurrently with CPU metering without
    /// borrowing the builder across threads.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn rvr_cuda_device_prewarm_task(
        &self,
    ) -> Option<Box<dyn FnOnce() -> Result<(), String> + Send + 'static>> {
        None
    }

    /// CUDA/G2: seal the default async allocation pool at the size reached by
    /// the real-shape warm pass and report its resulting counters.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn finish_rvr_cuda_device_prewarm(&self, _reserve_bytes: usize) -> Result<(), String> {
        Ok(())
    }

    /// Profiling-only snapshot of the default CUDA async allocation pool:
    /// reserved current/high, used current/high, and release threshold.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn rvr_cuda_device_pool_stats(&self) -> Result<Option<[u64; 5]>, String> {
        Ok(None)
    }

    /// CUDA/G2: return unused pages from the default async allocation pool at
    /// the decode-to-prove boundary while retaining a small warm floor.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn trim_rvr_cuda_device_pool(
        &self,
        _retain_bytes: usize,
        _segment_idx: usize,
    ) -> Result<(), String> {
        Ok(())
    }

    /// CUDA/G2: release the current segment's device trace-source bundle once
    /// trace generation and device continuation-state merging are complete.
    ///
    /// GPU builders with G2 state override this hook. Executable-wide device
    /// caches and CUDA module state must remain bound for later segments.
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    fn release_rvr_cuda_device_trace_sources(&self) {}

    /// Build the registry used by rvr preflight routing and record assembly.
    ///
    /// A composed builder adds its inner config's registrations first, then
    /// calls [`crate::arch::rvr::VmRvrLogNativeExtension::extend_rvr_log_native`]
    /// for each extension it owns.
    #[cfg(feature = "rvr")]
    #[allow(unused_variables)]
    fn create_rvr_log_native_assembler_registry(
        &self,
        config: &Self::VmConfig,
    ) -> LogNativeAssemblerRegistry<Val<E::SC>, Self::RecordArena>
    where
        Val<E::SC>: PrimeField32,
    {
        LogNativeAssemblerRegistry::new()
    }

    /// G2: airs whose inline records the proving path should stage as compact
    /// WIRE write targets — the generated C writes packed wire records
    /// straight into the arena backing that
    /// [`Self::generate_rvr_record_arenas_from_logs`]'s result hands to the
    /// chips (one alloc, no copy; the arena is marked wire-mode). Default:
    /// none. Only meaningful for dense record arenas whose chips can consume
    /// wire records (the GPU compact-decode path); a requested air must be
    /// compiled compact (NOT arena-native) — the proving path errors loudly
    /// on a mismatch instead of silently measuring the wrong emission.
    #[cfg(feature = "rvr")]
    #[allow(unused_variables)]
    fn rvr_wire_record_airs(
        &self,
        config: &Self::VmConfig,
        exe: &VmExe<Val<E::SC>>,
        pc_to_air_idx: &[Option<usize>],
        inline_meta: &crate::arch::rvr::RvrInlineRecordsMeta,
    ) -> std::collections::HashSet<usize>
    where
        Val<E::SC>: PrimeField32,
    {
        Default::default()
    }

    /// Optional rvr log-native record assembly hook.
    ///
    /// Builders normally customize this by overriding
    /// [`Self::create_rvr_log_native_assembler_registry`] and composing the
    /// registrations from their inner config plus their own extensions. An
    /// empty registry returns `None`; the proving path reports that as an
    /// explicit unsupported error rather than falling back to interpreter
    /// records after rvr execution.
    #[cfg(feature = "rvr")]
    #[allow(unused_variables)]
    fn generate_rvr_record_arenas_from_logs(
        &self,
        config: &Self::VmConfig,
        exe: &VmExe<Val<E::SC>>,
        output: &mut RvrPreflightOutput<Val<E::SC>>,
        capacities: &[(usize, usize)],
        pc_to_air_idx: &[Option<usize>],
    ) -> Result<Option<Vec<Self::RecordArena>>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
    {
        let registry = self.create_rvr_log_native_assembler_registry(config);
        if registry.is_empty() {
            return Ok(None);
        }
        generate_record_arenas_from_logs(&registry, exe, output, capacities, pc_to_air_idx)
            .map(Some)
    }
}

impl<SC, VC> VmConfig<SC> for VC
where
    SC: StarkProtocolConfig,
    VC: Clone
        + Serialize
        + DeserializeOwned
        + InitFileGenerator
        + VmExecutionConfig<Val<SC>>
        + VmCircuitConfig<SC>
        + AsRef<SystemConfig>
        + AsMut<SystemConfig>,
{
}

/// Trait for generating a init.rs file that contains a call to moduli_init!,
/// complex_init!, sw_init! with the supported moduli and curves.
/// Should be implemented by all VM config structs.
pub trait InitFileGenerator {
    // Default implementation is no init file.
    fn generate_init_file_contents(&self) -> Option<String> {
        None
    }

    // Do not override this method's default implementation.
    // This method is called by cargo openvm and the SDK before building the guest package.
    fn write_to_init_file(
        &self,
        manifest_dir: &Path,
        init_file_name: Option<&str>,
    ) -> io::Result<()> {
        if let Some(contents) = self.generate_init_file_contents() {
            let dest_path = Path::new(manifest_dir)
                .join(init_file_name.unwrap_or(OPENVM_DEFAULT_INIT_FILE_NAME));
            let mut f = File::create(&dest_path)?;
            write!(f, "{contents}")?;
        }
        Ok(())
    }
}

/// Each address space in guest memory may be configured with a different type `T` to represent a
/// memory cell in the address space. On host, the address space will be mapped to linear host
/// memory in bytes. The type `T` must be plain old data (POD) and be safely transmutable from a
/// fixed size array of bytes. Moreover, each type `T` must be convertible to a field element `F`.
///
/// We currently implement this trait on the enum [MemoryCellType], which includes all cell types
/// that we expect to be used in the VM context.
pub trait AddressSpaceHostLayout {
    /// Size in bytes of the memory cell type.
    fn size(&self) -> usize;

    /// # Safety
    /// - This function must only be called when `value` is guaranteed to be of size `self.size()`.
    /// - For `F`-cell layouts, `value` must be aligned for `F` and contain a valid `F`.
    unsafe fn to_field<F: Field>(&self, value: &[u8]) -> F;
}

#[derive(Debug, Serialize, Deserialize, Clone, new)]
pub struct MemoryConfig {
    /// The maximum height of the address space. This means the trie has `addr_space_height` layers
    /// for searching the address space. The allowed address spaces are those in the range `[1,
    /// 1 + 2^addr_space_height)` where it starts from 1 to not allow address space 0 in memory.
    pub addr_space_height: usize,
    /// It is expected that the size of the list is `(1 << addr_space_height) + 1` and the first
    /// element is 0, which means no address space.
    pub addr_spaces: Vec<AddressSpaceHostConfig>,
    /// Maximum bit width of AS-native OpenVM memory pointers.
    pub pointer_max_bits: usize,
    /// All timestamps must be in the range `[0, 2^timestamp_max_bits)`. Maximum allowed: 29.
    pub timestamp_max_bits: usize,
    /// Limb size used by the range checker
    pub decomp: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut addr_spaces =
            Self::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
        // RV64 register, memory, and public-values address spaces use u16 storage cells.
        addr_spaces[RV64_REGISTER_AS as usize].num_cells =
            NUM_RV64_REGISTERS * size_of::<u64>() / U16_CELL_SIZE;
        addr_spaces[RV64_MEMORY_AS as usize].num_cells = MEM_SIZE / U16_CELL_SIZE;
        addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
        addr_spaces[DEFERRAL_AS as usize].num_cells = DEFAULT_DEFERRAL_ADDR_SPACE_CELLS;
        Self::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17)
    }
}

impl MemoryConfig {
    pub fn empty_address_space_configs(num_addr_spaces: usize) -> Vec<AddressSpaceHostConfig> {
        // By default only address spaces 1..=4 have non-empty cell counts.
        let mut addr_spaces =
            vec![AddressSpaceHostConfig::new(0, MemoryCellType::field32()); num_addr_spaces];
        addr_spaces[RV64_IMM_AS as usize] = AddressSpaceHostConfig::new(0, MemoryCellType::Null);
        addr_spaces[RV64_REGISTER_AS as usize] =
            AddressSpaceHostConfig::new(0, MemoryCellType::U16);

        addr_spaces[RV64_MEMORY_AS as usize] = AddressSpaceHostConfig::new(0, MemoryCellType::U16);

        addr_spaces[PUBLIC_VALUES_AS as usize] =
            AddressSpaceHostConfig::new(0, MemoryCellType::U16);

        addr_spaces
    }

    /// Config for aggregation usage with only native address space.
    pub fn aggregation() -> Self {
        let mut addr_spaces =
            Self::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
        addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 28;
        Self::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17)
    }
}

/// System-level configuration for the virtual machine. Contains all configuration parameters that
/// are managed by the architecture.
#[derive(Debug, Clone, Serialize, Deserialize, Setters, WithSetters)]
pub struct SystemConfig {
    /// The maximum constraint degree any chip is allowed to use.
    #[getset(set_with = "pub")]
    pub max_constraint_degree: usize,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// Number of cells in the user public-values address space.
    pub num_public_values: usize,
    /// Max memory in bytes used across all chips for triggering segmentation.
    /// This field is skipped in serde as it's only used in execution and
    /// not needed after any serialize/deserialize.
    #[serde(skip, default = "default_segmentation_max_memory")]
    #[getset(set = "pub")]
    pub segmentation_max_memory: usize,
}

impl SystemConfig {
    pub fn new(
        max_constraint_degree: usize,
        mut memory_config: MemoryConfig,
        num_public_values: usize,
    ) -> Self {
        assert!(
            memory_config.timestamp_max_bits <= 29,
            "Timestamp max bits must be <= 29 for LessThan to work in 31-bit field"
        );
        assert_public_values_shape::<VM_DIGEST_WIDTH>(num_public_values);
        memory_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        Self {
            max_constraint_degree,
            memory_config,
            num_public_values,
            segmentation_max_memory: DEFAULT_MAX_MEMORY,
        }
    }

    pub fn default_from_memory(memory_config: MemoryConfig) -> Self {
        Self::new(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            memory_config,
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
        )
    }

    pub fn with_public_values(mut self, num_public_values: usize) -> Self {
        assert_public_values_shape::<VM_DIGEST_WIDTH>(num_public_values);
        self.num_public_values = num_public_values;
        self.memory_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        self
    }

    pub fn with_public_values_bytes(self, num_public_values_bytes: usize) -> Self {
        self.with_public_values(public_values_cells_from_bytes(num_public_values_bytes))
    }

    /// Returns the AIR ID of the memory boundary AIR. Panic if the boundary AIR is not enabled.
    pub fn memory_boundary_air_id(&self) -> usize {
        BOUNDARY_AIR_ID
    }

    /// Returns the AIR ID of the memory merkle AIR.
    pub fn memory_merkle_air_id(&self) -> usize {
        self.memory_boundary_air_id() + 1
    }

    /// Whether the AIR ID must be present in a valid v2 proof.
    pub fn is_required_air_id(&self, air_id: usize) -> bool {
        air_id == PROGRAM_AIR_ID
            || air_id == CONNECTOR_AIR_ID
            || air_id == self.memory_boundary_air_id()
            || air_id == self.memory_merkle_air_id()
    }

    /// This is O(1) and returns the length of
    /// [`SystemAirInventory::into_airs`](crate::system::SystemAirInventory::into_airs).
    pub fn num_airs(&self) -> usize {
        self.memory_boundary_air_id() + num_memory_airs()
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self::default_from_memory(MemoryConfig::default())
    }
}

impl AsRef<SystemConfig> for SystemConfig {
    fn as_ref(&self) -> &SystemConfig {
        self
    }
}

impl AsMut<SystemConfig> for SystemConfig {
    fn as_mut(&mut self) -> &mut SystemConfig {
        self
    }
}

// Default implementation uses no init file
impl InitFileGenerator for SystemConfig {}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, new)]
pub struct AddressSpaceHostConfig {
    /// The number of memory cells in each address space, where a memory cell refers to a single
    /// addressable unit of memory as defined by the ISA.
    pub num_cells: usize,
    pub layout: MemoryCellType,
}

impl AddressSpaceHostConfig {
    /// The total size in bytes of the address space in a linear memory layout.
    pub fn size(&self) -> usize {
        self.num_cells * self.layout.size()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCellType {
    Null,
    U8,
    /// U16 cells are stored as two little-endian bytes in linear byte storage:
    /// cell `k` is decoded from `bytes[2 * k]` and `bytes[2 * k + 1]`.
    U16,
    /// U32 cells are stored as four little-endian bytes in linear byte storage.
    U32,
    /// `size` is the size in bytes of the native field type. This should not exceed 8.
    F {
        size: u8,
    },
}

impl MemoryCellType {
    pub fn field32() -> Self {
        Self::F {
            size: size_of::<u32>() as u8,
        }
    }
}

impl AddressSpaceHostLayout for MemoryCellType {
    fn size(&self) -> usize {
        match self {
            Self::Null => 1, // to avoid divide by zero
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::F { size } => *size as usize,
        }
    }

    /// # Safety
    /// - This function must only be called when `value` is guaranteed to be of size `self.size()`.
    /// - For `F` cells, `value` must be aligned for `F` and contain a valid `F`.
    ///
    /// # Panics
    /// If the value is of integer type and overflows the field.
    unsafe fn to_field<F: Field>(&self, value: &[u8]) -> F {
        match self {
            Self::Null => unreachable!(),
            Self::U8 => F::from_u8(*value.get_unchecked(0)),
            Self::U16 => F::from_u16(u16::from_le_bytes([
                *value.get_unchecked(0),
                *value.get_unchecked(1),
            ])),
            Self::U32 => F::from_u32(u32::from_le_bytes([
                *value.get_unchecked(0),
                *value.get_unchecked(1),
                *value.get_unchecked(2),
                *value.get_unchecked(3),
            ])),
            Self::F { .. } => core::ptr::read(value.as_ptr() as *const F),
        }
    }
}
