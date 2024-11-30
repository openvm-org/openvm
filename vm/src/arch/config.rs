use std::collections::BTreeMap;

use ax_poseidon2_air::poseidon2::Poseidon2Config;
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
    ChipUsageGetter,
};
use axvm_circuit::system::memory::MemoryTraceHeights;
use derive_new::new;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use super::{
    AnyEnum, InstructionExecutor, SystemComplex, SystemExecutor, SystemPeriphery, VmChipComplex,
    VmInventoryError, PUBLIC_VALUES_AIR_ID,
};
use crate::{
    arch::ExecutorName,
    // intrinsics::modular::{SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
    system::memory::BOUNDARY_AIR_OFFSET,
};

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

pub trait VmGenericConfig<F: PrimeField32> {
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
    pub as_offset: usize,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    /// Limb size used by the range checker
    pub decomp: usize,
    /// Maximum N AccessAdapter AIR to support.
    pub max_access_adapter_n: usize,
    /// If set, the height of the trace of boundary AIR(for volatile memory) will be overridden.
    // TODO: remove this because we have MemoryTraceHeights
    pub boundary_air_height: Option<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new(29, 1, 29, 29, 17, 64, None)
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
    /// If set, the height of the traces will be overridden.
    pub overridden_heights: Option<SystemTraceHeights>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
            overridden_heights: None,
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
            0,
        )
    }
}

impl<F: PrimeField32> VmGenericConfig<F> for SystemConfig {
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

// to be deleted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    /// List of all executors except modular executors.
    pub executors: Vec<ExecutorName>,
    /// Optional. Can be used to override the height of the trace of an executor.
    pub overridden_executor_heights: Option<BTreeMap<ExecutorName, usize>>,
    /// List of all supported modulus
    pub supported_modulus: Vec<BigUint>,
    /// List of all supported Complex extensions, stored as indices of supported_modulus.
    /// The supported modulus must exist in order for the complex extension to be supported.
    pub supported_complex_ext: Vec<usize>,

    pub poseidon2_max_constraint_degree: usize,
    /// True if the VM is in continuation mode. In this mode, an execution could be segmented and
    /// each segment is proved by a proof. Each proof commits the before and after state of the
    /// corresponding segment.
    /// False if the VM is in single segment mode. In this mode, an execution is proved by a single
    /// proof.
    pub continuation_enabled: bool,
    pub memory_config: MemoryConfig,
    /// `num_public_values` has different meanings in single segment mode and continuation mode.
    /// In single segment mode, `num_public_values` is the number of public values of
    /// PublicValuesChips. In this case, verifier can read public values directly.
    /// In continuation mode, public values are stored in a special address space.
    /// `number_public_values` indicates the number of allowed addresses in that address space. The verifier
    /// cannot read public values directly, but they can decommit the public values from the memory
    /// state commit.
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
}

impl VmConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parameters(
        poseidon2_max_constraint_degree: usize,
        continuation_enabled: bool,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        // Come from CompilerOptions. We can also pass in the whole compiler option if we need more fields from it.
        supported_modulus: Vec<BigUint>,
        supported_complex_ext: Vec<usize>,
    ) -> Self {
        VmConfig {
            executors: Vec::new(),
            overridden_executor_heights: None,
            continuation_enabled,
            poseidon2_max_constraint_degree,
            memory_config,
            num_public_values,
            max_segment_len,
            collect_metrics,
            supported_modulus,
            supported_complex_ext,
        }
    }

    pub fn add_executor(mut self, executor: ExecutorName) -> Self {
        // Some executors need to be handled in a special way, and cannot be added like other executors.
        // Adding these will cause a panic in the `create_chip_set` function.
        self.executors.push(executor);
        self
    }

    pub fn with_num_public_values(mut self, n: usize) -> Self {
        self.num_public_values = n;
        self
    }

    pub fn with_max_segment_len(mut self, n: usize) -> Self {
        self.max_segment_len = n;
        self
    }

    /// Generate a proving key for the VM.
    pub fn generate_pk<SC: StarkGenericConfig>(
        &self,
        mut keygen_builder: MultiStarkKeygenBuilder<SC>,
    ) -> MultiStarkProvingKey<SC>
    where
        Val<SC>: PrimeField32,
    {
        let chip_set = self.create_chip_set::<Val<SC>>();
        for air in chip_set.airs() {
            keygen_builder.add_air(air);
        }
        keygen_builder.generate_pk()
    }
}

impl Default for VmConfig {
    fn default() -> Self {
        Self::from_parameters(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            false,
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            vec![],
            vec![],
        )
    }
}

impl VmConfig {
    pub fn rv32i() -> Self {
        VmConfig {
            poseidon2_max_constraint_degree: 3,
            continuation_enabled: true,
            ..Default::default()
        }
        .add_executor(ExecutorName::Phantom)
        .add_executor(ExecutorName::BaseAluRv32)
        .add_executor(ExecutorName::LessThanRv32)
        .add_executor(ExecutorName::ShiftRv32)
        .add_executor(ExecutorName::LoadStoreRv32)
        .add_executor(ExecutorName::LoadSignExtendRv32)
        .add_executor(ExecutorName::HintStoreRv32)
        .add_executor(ExecutorName::BranchEqualRv32)
        .add_executor(ExecutorName::BranchLessThanRv32)
        .add_executor(ExecutorName::JalLuiRv32)
        .add_executor(ExecutorName::JalrRv32)
        .add_executor(ExecutorName::AuipcRv32)
    }

    pub fn rv32im() -> Self {
        Self::rv32i()
            .add_executor(ExecutorName::MultiplicationRv32)
            .add_executor(ExecutorName::MultiplicationHighRv32)
            .add_executor(ExecutorName::DivRemRv32)
    }

    pub fn read_config_file(file: &str) -> Result<Self, String> {
        let file_str = std::fs::read_to_string(file)
            .map_err(|_| format!("Could not load config file from: {file}"))?;
        let config: Self = toml::from_str(file_str.as_str())
            .map_err(|e| format!("Failed to parse config file {}:\n{}", file, e))?;
        Ok(config)
    }
}
