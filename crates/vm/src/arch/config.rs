use std::{fs::File, io::Write, path::Path, sync::Arc};

use derive_new::new;
use openvm_circuit::system::memory::MemoryTraceHeights;
use openvm_instructions::NATIVE_AS;
use openvm_poseidon2_air::Poseidon2Config;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    p3_field::Field,
    prover::hal::ProverBackend,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::{
    segmentation_strategy::{DefaultSegmentationStrategy, SegmentationStrategy},
    AnyEnum, VmChipComplex, PUBLIC_VALUES_AIR_ID,
};
use crate::{
    arch::{
        AirInventory, AirInventoryError, Arena, ChipInventoryError, ExecutorInventory,
        ExecutorInventoryError,
    },
    system::{
        memory::{
            merkle::public_values::PUBLIC_VALUES_AS, num_memory_airs, BOUNDARY_AIR_OFFSET, CHUNK,
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
///
/// For users who only need to create an execution environment, use the sub-trait
/// [VmExecutionConfig] to avoid the `SC` generic.
///
/// This trait does not contain the [VmProverConfig] trait, because a single VM configuration may
/// implement multiple [VmProverConfig]s for different prover backends.
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
    type Executor: AnyEnum;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;
}

pub trait VmCircuitConfig<SC: StarkGenericConfig> {
    fn create_circuit(&self) -> Result<AirInventory<SC>, AirInventoryError>;

    /// Generate the proving key and verifying key for the circuit defined by this config.
    fn keygen(&self, stark_config: &SC) -> Result<MultiStarkProvingKey<SC>, AirInventoryError> {
        let circuit = self.create_circuit()?;
        let pk = circuit.keygen(stark_config);
        Ok(pk)
    }
}

pub trait VmProverConfig<SC, PB>: VmConfig<SC>
where
    SC: StarkGenericConfig,
    PB: ProverBackend<Val = Val<SC>, Challenge = SC::Challenge, Challenger = SC::Challenger>,
{
    type RecordArena: Arena;
    type SystemChipInventory: SystemChipComplex<Self::RecordArena, PB>;

    /// Create a [VmChipComplex] from the full [AirInventory], which should be the output of
    /// [VmCircuitConfig::create_circuit].
    fn create_chip_complex(
        &self,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, PB, Self::SystemChipInventory>,
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
    ) -> eyre::Result<()> {
        if let Some(contents) = self.generate_init_file_contents() {
            let dest_path = Path::new(manifest_dir)
                .join(init_file_name.unwrap_or(OPENVM_DEFAULT_INIT_FILE_NAME));
            let mut f = File::create(&dest_path)?;
            write!(f, "{}", contents)?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, new)]
pub struct MemoryConfig {
    /// The maximum height of the address space. This means the trie has `addr_space_height` layers
    /// for searching the address space. The allowed address spaces are those in the range `[1,
    /// 1 + 2^addr_space_height)` where it starts from 1 to not allow address space 0 in memory.
    pub addr_space_height: usize,
    /// The number of cells in each address space. It is expected that the size of the list is
    /// `1 << addr_space_height + 1` and the first element is 0, which means no address space.
    pub addr_space_sizes: Vec<usize>,
    pub pointer_max_bits: usize,
    /// All timestamps must be in the range `[0, 2^clk_max_bits)`. Maximum allowed: 29.
    pub clk_max_bits: usize,
    /// Limb size used by the range checker
    pub decomp: usize,
    /// Maximum N AccessAdapter AIR to support.
    pub max_access_adapter_n: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let mut addr_space_sizes = vec![0; (1 << 3) + ADDR_SPACE_OFFSET as usize];
        addr_space_sizes[ADDR_SPACE_OFFSET as usize..=NATIVE_AS as usize].fill(1 << 29);
        addr_space_sizes[PUBLIC_VALUES_AS as usize] = DEFAULT_MAX_NUM_PUBLIC_VALUES;
        Self::new(3, addr_space_sizes, 29, 29, 17, 32)
    }
}

impl MemoryConfig {
    /// Config for aggregation usage with only native address space.
    pub fn aggregation() -> Self {
        let mut addr_space_sizes = vec![0; (1 << 3) + ADDR_SPACE_OFFSET as usize];
        addr_space_sizes[NATIVE_AS as usize] = 1 << 29;
        Self::new(3, addr_space_sizes, 29, 29, 17, 8)
    }
}

/// System-level configuration for the virtual machine. Contains all configuration parameters that
/// are managed by the architecture, including configuration for continuations support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// The maximum constraint degree any chip is allowed to use.
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
    /// Segmentation strategy
    /// This field is skipped in serde as it's only used in execution and
    /// not needed after any serialize/deserialize.
    #[serde(skip, default = "get_default_segmentation_strategy")]
    pub segmentation_strategy: Arc<dyn SegmentationStrategy>,
}

pub fn get_default_segmentation_strategy() -> Arc<DefaultSegmentationStrategy> {
    Arc::new(DefaultSegmentationStrategy::default())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemTraceHeights {
    pub memory: MemoryTraceHeights,
    // All other chips have constant heights.
}

impl SystemConfig {
    pub fn new(
        max_constraint_degree: usize,
        mut memory_config: MemoryConfig,
        num_public_values: usize,
    ) -> Self {
        let segmentation_strategy = get_default_segmentation_strategy();
        assert!(
            memory_config.clk_max_bits <= 29,
            "Timestamp max bits must be <= 29 for LessThan to work in 31-bit field"
        );
        memory_config.addr_space_sizes[PUBLIC_VALUES_AS as usize] = num_public_values;
        Self {
            max_constraint_degree,
            continuation_enabled: false,
            memory_config,
            num_public_values,
            segmentation_strategy,
            profiling: false,
        }
    }

    pub fn default_from_memory(memory_config: MemoryConfig) -> Self {
        Self::new(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            memory_config,
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
        )
    }

    pub fn with_max_constraint_degree(mut self, max_constraint_degree: usize) -> Self {
        self.max_constraint_degree = max_constraint_degree;
        self
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
        self.memory_config.addr_space_sizes[PUBLIC_VALUES_AS as usize] = num_public_values;
        self
    }

    pub fn with_max_segment_len(mut self, max_segment_len: usize) -> Self {
        self.segmentation_strategy = Arc::new(
            DefaultSegmentationStrategy::new_with_max_segment_len(max_segment_len),
        );
        self
    }

    pub fn set_segmentation_strategy(&mut self, strategy: Arc<dyn SegmentationStrategy>) {
        self.segmentation_strategy = strategy;
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
        let mut ret = PUBLIC_VALUES_AIR_ID;
        if self.has_public_values_chip() {
            ret += 1;
        }
        ret += BOUNDARY_AIR_OFFSET;
        ret
    }

    /// This is O(1) and returns the length of
    /// [`SystemAirInventory::into_airs`](crate::system::SystemAirInventory::into_airs).
    pub fn num_airs(&self) -> usize {
        2 + usize::from(self.has_public_values_chip())
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

impl SystemTraceHeights {
    /// Round all trace heights to the next power of two. This will round trace heights of 0 to 1.
    pub fn round_to_next_power_of_two(&mut self) {
        self.memory.round_to_next_power_of_two();
    }

    /// Round all trace heights to the next power of two, except 0 stays 0.
    pub fn round_to_next_power_of_two_or_zero(&mut self) {
        self.memory.round_to_next_power_of_two_or_zero();
    }
}

// Default implementation uses no init file
impl InitFileGenerator for SystemConfig {}
