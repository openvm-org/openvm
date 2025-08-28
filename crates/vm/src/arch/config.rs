use std::{
    fs::File,
    io::{self, Write},
    path::Path,
};

use derive_new::new;
use getset::{Setters, WithSetters};
use openvm_instructions::{
    riscv::{RV32_IMM_AS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    NATIVE_AS,
};
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::Field,
    p3_util::log2_strict_usize,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{AnyEnum, VmChipComplex, PUBLIC_VALUES_AIR_ID};
use crate::{
    arch::{
        execution_mode::metered::segment_ctx::SegmentationLimits, AirInventory, AirInventoryError,
        Arena, ChipInventoryError, ExecutorInventory, ExecutorInventoryError,
    },
    system::{
        memory::{
            merkle::public_values::PUBLIC_VALUES_AS, num_memory_airs, CHUNK, POINTER_MAX_BITS,
        },
        SystemChipComplex,
    },
};

// sbox is decomposed to have this max degree for Poseidon2. We set to 3 so quotient_degree = 2
// allows log_blowup = 1
const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 3;
pub const DEFAULT_MAX_NUM_PUBLIC_VALUES: usize = 32;
/// Width of Poseidon2 VM uses.
pub const POSEIDON2_WIDTH: usize = 16;
/// Offset for address space indices. This is used to distinguish between different memory spaces.
pub const ADDR_SPACE_OFFSET: u32 = 1;
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
    SC: StarkGenericConfig,
{
}

pub trait VmExecutionConfig<F> {
    type Executor: AnyEnum + Send + Sync;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;
}

pub trait VmCircuitConfig<SC: StarkGenericConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}

/// This trait is intended to be implemented on a new type wrapper of the VmConfig struct to get
/// around Rust orphan rules.
pub trait VmBuilder<E: StarkEngine>: Sized {
    type VmConfig: VmConfig<E::SC>;
    type RecordArena: Arena;
    type SystemChipInventory: SystemChipComplex<Self::RecordArena, E::PB>;

    /// Create a [VmChipComplex] from the full [AirInventory], which should be the output of
    /// [VmCircuitConfig::create_airs].
    #[allow(clippy::type_complexity)]
    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<E::SC>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    >;
}

impl<SC, VC> VmConfig<SC> for VC
where
    SC: StarkGenericConfig,
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

pub const OPENVM_DEFAULT_INIT_FILE_BASENAME: &str = "openvm_init";
pub const OPENVM_DEFAULT_INIT_FILE_NAME: &str = "openvm_init.rs";
/// The minimum block size is 4, but RISC-V `lb` only requires alignment of 1 and `lh` only requires
/// alignment of 2 because the instructions are implemented by doing an access of block size 4.
const DEFAULT_U8_BLOCK_SIZE: usize = 4;
const DEFAULT_NATIVE_BLOCK_SIZE: usize = 1;

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
            write!(f, "{}", contents)?;
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
    /// - Alignment of `value` must be a multiple of the alignment of `F`.
    /// - The field type `F` must be plain old data.
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
    pub pointer_max_bits: usize,
    /// All timestamps must be in the range `[0, 2^timestamp_max_bits)`. Maximum allowed: 29.
    pub timestamp_max_bits: usize,
    /// Limb size used by the range checker
    pub decomp: usize,
    /// Maximum N AccessAdapter AIR to support.
    pub max_access_adapter_n: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut addr_spaces =
            Self::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
        const MAX_CELLS: usize = 1 << 29;
        addr_spaces[RV32_REGISTER_AS as usize].num_cells = 32 * size_of::<u32>();
        addr_spaces[RV32_MEMORY_AS as usize].num_cells = MAX_CELLS;
        addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = DEFAULT_MAX_NUM_PUBLIC_VALUES;
        addr_spaces[NATIVE_AS as usize].num_cells = MAX_CELLS;
        Self::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17, 32)
    }
}

impl MemoryConfig {
    pub fn empty_address_space_configs(num_addr_spaces: usize) -> Vec<AddressSpaceHostConfig> {
        // All except address spaces 0..4 default to native 32-bit field.
        // By default only address spaces 1..=4 have non-empty cell counts.
        let mut addr_spaces = vec![
            AddressSpaceHostConfig::new(
                0,
                DEFAULT_NATIVE_BLOCK_SIZE,
                MemoryCellType::native32()
            );
            num_addr_spaces
        ];
        addr_spaces[RV32_IMM_AS as usize] = AddressSpaceHostConfig::new(0, 1, MemoryCellType::Null);
        addr_spaces[RV32_REGISTER_AS as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);
        addr_spaces[RV32_MEMORY_AS as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);
        addr_spaces[PUBLIC_VALUES_AS as usize] =
            AddressSpaceHostConfig::new(0, DEFAULT_U8_BLOCK_SIZE, MemoryCellType::U8);

        addr_spaces
    }

    /// Config for aggregation usage with only native address space.
    pub fn aggregation() -> Self {
        let mut addr_spaces =
            Self::empty_address_space_configs((1 << 3) + ADDR_SPACE_OFFSET as usize);
        addr_spaces[NATIVE_AS as usize].num_cells = 1 << 29;
        Self::new(3, addr_spaces, POINTER_MAX_BITS, 29, 17, 8)
    }

    pub fn min_block_size_bits(&self) -> Vec<u8> {
        self.addr_spaces
            .iter()
            .map(|addr_sp| log2_strict_usize(addr_sp.min_block_size) as u8)
            .collect()
    }
}

/// System-level configuration for the virtual machine. Contains all configuration parameters that
/// are managed by the architecture, including configuration for continuations support.
#[derive(Debug, Clone, Serialize, Deserialize, Setters, WithSetters)]
pub struct SystemConfig {
    /// The maximum constraint degree any chip is allowed to use.
    #[getset(set_with = "pub")]
    pub max_constraint_degree: usize,
    /// True if the VM is in continuation mode. In this mode, an execution could be segmented and
    /// each segment is proved by a proof. Each proof commits the before and after state of the
    /// corresponding segment.
    /// False if the VM is in single segment mode. In this mode, an execution is proved by a single
    /// proof.
    pub continuation_enabled: bool,
    /// Memory configuration
    pub memory_config: MemoryConfig,
    /// `num_public_values` has different meanings in single segment mode and continuation mode.
    /// In single segment mode, `num_public_values` is the number of public values of
    /// `PublicValuesChip`. In this case, verifier can read public values directly.
    /// In continuation mode, public values are stored in a special address space.
    /// `num_public_values` indicates the number of allowed addresses in that address space. The
    /// verifier cannot read public values directly, but they can decommit the public values
    /// from the memory merkle root.
    pub num_public_values: usize,
    /// Whether to collect detailed profiling metrics.
    /// **Warning**: this slows down the runtime.
    pub profiling: bool,
    /// Segmentation limits
    /// This field is skipped in serde as it's only used in execution and
    /// not needed after any serialize/deserialize.
    #[serde(skip, default = "SegmentationLimits::default")]
    #[getset(set = "pub")]
    pub segmentation_limits: SegmentationLimits,
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
        memory_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        Self {
            max_constraint_degree,
            continuation_enabled: true,
            memory_config,
            num_public_values,
            profiling: false,
            segmentation_limits: SegmentationLimits::default(),
        }
    }

    pub fn default_from_memory(memory_config: MemoryConfig) -> Self {
        Self::new(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            memory_config,
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
        )
    }

    pub fn with_continuations(mut self) -> Self {
        self.continuation_enabled = true;
        self
    }

    pub fn without_continuations(mut self) -> Self {
        self.continuation_enabled = false;
        self
    }

    pub fn with_public_values(mut self, num_public_values: usize) -> Self {
        self.num_public_values = num_public_values;
        self.memory_config.addr_spaces[PUBLIC_VALUES_AS as usize].num_cells = num_public_values;
        self
    }

    pub fn with_max_segment_len(mut self, max_segment_len: usize) -> Self {
        self.segmentation_limits.max_trace_height = max_segment_len as u32;
        self
    }

    pub fn with_profiling(mut self) -> Self {
        self.profiling = true;
        self
    }

    pub fn without_profiling(mut self) -> Self {
        self.profiling = false;
        self
    }

    pub fn has_public_values_chip(&self) -> bool {
        !self.continuation_enabled && self.num_public_values > 0
    }

    /// Returns the AIR ID of the memory boundary AIR. Panic if the boundary AIR is not enabled.
    pub fn memory_boundary_air_id(&self) -> usize {
        PUBLIC_VALUES_AIR_ID + usize::from(self.has_public_values_chip())
    }

    /// Returns the AIR ID of the memory merkle AIR. Returns None if continuations are not enabled.
    pub fn memory_merkle_air_id(&self) -> Option<usize> {
        let boundary_idx = self.memory_boundary_air_id();
        if self.continuation_enabled {
            Some(boundary_idx + 1)
        } else {
            None
        }
    }

    /// AIR ID for the first memory access adapter AIR.
    pub fn access_adapter_air_id_offset(&self) -> usize {
        let boundary_idx = self.memory_boundary_air_id();
        // boundary, (if persistent memory) merkle AIRs
        boundary_idx + 1 + usize::from(self.continuation_enabled)
    }

    /// This is O(1) and returns the length of
    /// [`SystemAirInventory::into_airs`](crate::system::SystemAirInventory::into_airs).
    pub fn num_airs(&self) -> usize {
        self.memory_boundary_air_id()
            + num_memory_airs(
                self.continuation_enabled,
                self.memory_config.max_access_adapter_n,
            )
    }

    pub fn initial_block_size(&self) -> usize {
        match self.continuation_enabled {
            true => CHUNK,
            false => 1,
        }
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
    /// Minimum block size for memory accesses supported. This is a property of the address space
    /// that is determined by the ISA.
    ///
    /// **Note**: Block size is in terms of memory cells.
    pub min_block_size: usize,
    pub layout: MemoryCellType,
}

impl AddressSpaceHostConfig {
    /// The total size in bytes of the address space in a linear memory layout.
    pub fn size(&self) -> usize {
        self.num_cells * self.layout.size()
    }
}

pub(crate) const MAX_CELL_BYTE_SIZE: usize = 8;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCellType {
    Null,
    U8,
    U16,
    /// Represented in little-endian format.
    U32,
    /// `size` is the size in bytes of the native field type. This should not exceed 8.
    Native {
        size: u8,
    },
}

impl MemoryCellType {
    pub fn native32() -> Self {
        Self::Native {
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
            Self::Native { size } => *size as usize,
        }
    }

    /// # Safety
    /// - This function must only be called when `value` is guaranteed to be of size `self.size()`.
    /// - Alignment of `value` must be a multiple of the alignment of `F`.
    /// - The field type `F` must be plain old data.
    ///
    /// # Panics
    /// If the value is of integer type and overflows the field.
    unsafe fn to_field<F: Field>(&self, value: &[u8]) -> F {
        match self {
            Self::Null => unreachable!(),
            Self::U8 => F::from_canonical_u8(*value.get_unchecked(0)),
            Self::U16 => F::from_canonical_u16(core::ptr::read(value.as_ptr() as *const u16)),
            Self::U32 => F::from_canonical_u32(core::ptr::read(value.as_ptr() as *const u32)),
            Self::Native { .. } => core::ptr::read(value.as_ptr() as *const F),
        }
    }
}
