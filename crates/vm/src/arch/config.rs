use derive_new::new;
use openvm_circuit::system::memory::MemoryTraceHeights;
use openvm_instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_poseidon2_air::poseidon2::Poseidon2Config;
use openvm_stark_backend::{p3_field::PrimeField32, ChipUsageGetter};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// TODO[jpw]: re-exporting hardcoded bus constants for tests. Import paths should be
// updated directly but it changes many files.
#[cfg(any(test, feature = "test-utils"))]
pub use super::testing::{
    BITWISE_OP_LOOKUP_BUS, BYTE_XOR_BUS, EXECUTION_BUS, MEMORY_BUS, MEMORY_MERKLE_BUS,
    POSEIDON2_DIRECT_BUS, RANGE_TUPLE_CHECKER_BUS, READ_INSTRUCTION_BUS,
};
use super::{
    AnyEnum, InstructionExecutor, SystemComplex, SystemExecutor, SystemPeriphery, VmChipComplex,
    VmInventoryError, PUBLIC_VALUES_AIR_ID,
};
use crate::system::memory::BOUNDARY_AIR_OFFSET;

const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 22) - 100;
// sbox is decomposed to have this max degree for Poseidon2. We set to 3 so quotient_degree = 2
// allows log_blowup = 1
const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 3;
/// Width of Poseidon2 VM uses.
pub const POSEIDON2_WIDTH: usize = 16;
/// Returns a Poseidon2 config for the VM.
pub fn vm_poseidon2_config<F: PrimeField32>() -> Poseidon2Config<POSEIDON2_WIDTH, F> {
    Poseidon2Config::<POSEIDON2_WIDTH, F>::new_p3_baby_bear_16()
}

pub trait VmConfig<F: PrimeField32>: Clone + Serialize + DeserializeOwned {
    type Executor: InstructionExecutor<F> + AnyEnum + ChipUsageGetter;
    type Periphery: AnyEnum + ChipUsageGetter;

    /// Must contain system config
    fn system(&self) -> &SystemConfig;
    fn system_mut(&mut self) -> &mut SystemConfig;

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError>;
}

#[derive(Debug, Serialize, Deserialize, Clone, new, Copy)]
pub struct MemoryConfig {
    /// The maximum height of the address space. This means the trie has `as_height` layers for searching the address space. The allowed address spaces are those in the range `[as_offset, as_offset + 2^as_height)` where `as_offset` is currently fixed to `1` to not allow address space `0` in memory.
    pub as_height: usize,
    /// The offset of the address space.
    pub as_offset: u32,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    /// Limb size used by the range checker
    pub decomp: usize,
    /// Maximum N AccessAdapter AIR to support.
    pub max_access_adapter_n: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new(29, 1, 29, 29, 17, 64)
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
    /// `num_public_values` indicates the number of allowed addresses in that address space. The verifier
    /// cannot read public values directly, but they can decommit the public values from the memory
    /// merkle root.
    pub num_public_values: usize,
    /// When continuations are enabled, a heuristic used to determine when to segment execution.
    pub max_segment_len: usize,
    /// Whether to collect metrics.
    /// **Warning**: this slows down the runtime.
    pub collect_metrics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SystemTraceHeights {
    pub memory: MemoryTraceHeights,
    // All other chips have constant heights.
}

impl SystemConfig {
    pub fn new(
        max_constraint_degree: usize,
        memory_config: MemoryConfig,
        num_public_values: usize,
    ) -> Self {
        Self {
            max_constraint_degree,
            continuation_enabled: false,
            memory_config,
            num_public_values,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            collect_metrics: false,
        }
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
        self
    }

    pub fn with_max_segment_len(mut self, max_segment_len: usize) -> Self {
        self.max_segment_len = max_segment_len;
        self
    }

    pub fn with_metric_collection(mut self) -> Self {
        self.collect_metrics = true;
        self
    }

    pub fn without_metric_collection(mut self) -> Self {
        self.collect_metrics = false;
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
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            Default::default(),
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
        )
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

impl<F: PrimeField32> VmConfig<F> for SystemConfig {
    type Executor = SystemExecutor<F>;
    type Periphery = SystemPeriphery<F>;

    fn system(&self) -> &SystemConfig {
        self
    }
    fn system_mut(&mut self) -> &mut SystemConfig {
        self
    }

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError> {
        let complex = SystemComplex::new(self.clone());
        Ok(complex)
    }
}
