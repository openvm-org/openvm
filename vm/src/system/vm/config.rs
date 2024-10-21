use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    keygen::{types::MultiStarkProvingKey, MultiStarkKeygenBuilder},
};
use derive_new::new;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};

use crate::{
    arch::ExecutorName,
    intrinsics::modular::{SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
    kernels::core::CoreOptions,
};

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 25) - 100;
pub const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 7; // the sbox degree used for Poseidon2

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceType {
    Persistent,
    Volatile,
}

#[derive(Debug, Serialize, Deserialize, Clone, new, Copy)]
pub struct MemoryConfig {
    pub addr_space_max_bits: usize,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    pub decomp: usize,
    pub persistence_type: PersistenceType,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new(29, 29, 29, 15, PersistenceType::Volatile)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    /// List of all executors except modular executors.
    pub executors: Vec<ExecutorName>,
    /// List of all supported modulus
    pub supported_modulus: Vec<BigUint>,

    pub poseidon2_max_constraint_degree: usize,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
}

impl VmConfig {
    pub fn from_parameters(
        poseidon2_max_constraint_degree: usize,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        // Come from CompilerOptions. We can also pass in the whole compiler option if we need more fields from it.
        enabled_modulus: Vec<BigUint>,
    ) -> Self {
        let config = VmConfig {
            executors: Vec::new(),
            poseidon2_max_constraint_degree,
            memory_config,
            num_public_values,
            max_segment_len,
            collect_metrics,
            supported_modulus: Vec::new(),
        };
        config.add_modular_support(enabled_modulus)
    }

    pub fn add_default_executor(mut self, executor: ExecutorName) -> Self {
        // Some executors need to be handled in a special way, and cannot be added like other executors.
        // Adding these will cause a panic in the `create_chip_set` function.
        self.executors.push(executor);
        self
    }

    // I think adding "opcode class" support is better than adding "executor".
    // The api should be saying: I want to be able to do this set of operations, and doesn't care about what executor is doing it.
    pub fn add_modular_support(self, enabled_modulus: Vec<BigUint>) -> Self {
        let mut res = self;
        res.supported_modulus.extend(enabled_modulus);
        res
    }

    pub fn add_canonical_modulus(self) -> Self {
        let primes = Modulus::all().iter().map(|m| m.prime()).collect();
        self.add_modular_support(primes)
    }

    pub fn add_ecc_support(self) -> Self {
        todo!()
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
        Self::default_with_no_executors()
            .add_default_executor(ExecutorName::Core)
            .add_default_executor(ExecutorName::FieldArithmetic)
            .add_default_executor(ExecutorName::FieldExtension)
            .add_default_executor(ExecutorName::Poseidon2)
    }
}

impl VmConfig {
    pub fn default_with_no_executors() -> Self {
        Self::from_parameters(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            vec![],
        )
    }

    pub fn core_options(&self) -> CoreOptions {
        CoreOptions {
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        Self::from_parameters(
            DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE,
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            vec![],
        )
        .add_default_executor(ExecutorName::Core)
    }

    pub fn rv32() -> Self {
        Self::core()
            .add_default_executor(ExecutorName::ArithmeticLogicUnitRv32)
            .add_default_executor(ExecutorName::LessThanRv32)
            .add_default_executor(ExecutorName::MultiplicationRv32)
            .add_default_executor(ExecutorName::MultiplicationHighRv32)
            .add_default_executor(ExecutorName::DivRemRv32)
            .add_default_executor(ExecutorName::ShiftRv32)
            .add_default_executor(ExecutorName::LoadStoreRv32)
            .add_default_executor(ExecutorName::BranchEqualRv32)
            .add_default_executor(ExecutorName::BranchLessThanRv32)
            .add_default_executor(ExecutorName::JalLuiRv32)
            .add_default_executor(ExecutorName::JalrRv32)
            .add_default_executor(ExecutorName::AuipcRv32)
    }

    pub fn aggregation(poseidon2_max_constraint_degree: usize) -> Self {
        VmConfig {
            poseidon2_max_constraint_degree,
            num_public_values: 4,
            ..VmConfig::default()
        }
    }
}

impl VmConfig {
    pub fn read_config_file(file: &str) -> Result<Self, String> {
        let file_str = std::fs::read_to_string(file)
            .map_err(|_| format!("Could not load config file from: {file}"))?;
        let config: Self = toml::from_str(file_str.as_str())
            .map_err(|e| format!("Failed to parse config file {}:\n{}", file, e))?;
        Ok(config)
    }
}

// TO BE DELETED:
#[derive(EnumCount, EnumIter, FromRepr, Clone, Debug)]
#[repr(usize)]
pub enum Modulus {
    Secp256k1Coord = 0,
    Secp256k1Scalar = 1,
}

impl Modulus {
    pub fn prime(&self) -> BigUint {
        match self {
            Modulus::Secp256k1Coord => SECP256K1_COORD_PRIME.clone(),
            Modulus::Secp256k1Scalar => SECP256K1_SCALAR_PRIME.clone(),
        }
    }

    pub fn all() -> Vec<Self> {
        Modulus::iter().collect()
    }
}
