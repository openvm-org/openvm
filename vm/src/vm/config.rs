use std::ops::Range;

use derive_new::new;
use num_bigint_dig::BigUint;
use serde::{Deserialize, Serialize};
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};

use crate::{
    arch::{instructions::*, ExecutorName},
    core::CoreOptions,
    modular_addsub::{SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
};

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 25) - 100;
pub const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 7; // the sbox degree used for Poseidon2

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceType {
    Persistent,
    Volatile,
}

#[derive(Debug, Serialize, Deserialize, Clone, new)]
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

fn default_executor_range(executor: ExecutorName) -> (Range<usize>, usize) {
    let (start, len, offset) = match executor {
        ExecutorName::Core => (
            CoreOpcode::default_offset(),
            CoreOpcode::COUNT,
            CoreOpcode::default_offset(),
        ),
        ExecutorName::FieldArithmetic => (
            FieldArithmeticOpcode::default_offset(),
            FieldArithmeticOpcode::COUNT,
            FieldArithmeticOpcode::default_offset(),
        ),
        ExecutorName::FieldExtension => (
            FieldExtensionOpcode::default_offset(),
            FieldExtensionOpcode::COUNT,
            FieldExtensionOpcode::default_offset(),
        ),
        ExecutorName::Poseidon2 => (
            Poseidon2Opcode::default_offset(),
            Poseidon2Opcode::COUNT,
            Poseidon2Opcode::default_offset(),
        ),
        ExecutorName::Keccak256 => (
            Keccak256Opcode::default_offset(),
            Keccak256Opcode::COUNT,
            Keccak256Opcode::default_offset(),
        ),
        ExecutorName::ModularAddSub => (
            ModularArithmeticOpcode::default_offset(),
            2,
            ModularArithmeticOpcode::default_offset(),
        ),
        ExecutorName::ModularMultDiv => (
            ModularArithmeticOpcode::default_offset() + 2,
            2,
            ModularArithmeticOpcode::default_offset(),
        ),
        ExecutorName::ArithmeticLogicUnitRv32 => (
            AluOpcode::default_offset(),
            AluOpcode::COUNT,
            AluOpcode::default_offset(),
        ),
        ExecutorName::LoadStoreRv32 => (
            Rv32LoadStoreOpcode::default_offset(),
            Rv32LoadStoreOpcode::COUNT,
            Rv32LoadStoreOpcode::default_offset(),
        ),
        ExecutorName::JalLuiRv32 => (
            Rv32JalLuiOpcode::default_offset(),
            Rv32JalLuiOpcode::COUNT,
            Rv32JalLuiOpcode::default_offset(),
        ),
        ExecutorName::ArithmeticLogicUnit256 => (
            U256Opcode::default_offset(),
            8,
            U256Opcode::default_offset(),
        ),
        ExecutorName::LessThanRv32 => (
            LessThanOpcode::default_offset(),
            LessThanOpcode::COUNT,
            LessThanOpcode::default_offset(),
        ),
        ExecutorName::MultiplicationRv32 => (
            MulOpcode::default_offset(),
            MulOpcode::COUNT,
            MulOpcode::default_offset(),
        ),
        ExecutorName::MultiplicationHighRv32 => (
            MulHOpcode::default_offset(),
            MulHOpcode::COUNT,
            MulHOpcode::default_offset(),
        ),
        ExecutorName::U256Multiplication => (
            U256Opcode::default_offset() + 11,
            1,
            U256Opcode::default_offset(),
        ),
        ExecutorName::DivRemRv32 => (
            DivRemOpcode::default_offset(),
            DivRemOpcode::COUNT,
            DivRemOpcode::default_offset(),
        ),
        ExecutorName::ShiftRv32 => (
            ShiftOpcode::default_offset(),
            ShiftOpcode::COUNT,
            ShiftOpcode::default_offset(),
        ),
        ExecutorName::Shift256 => (
            U256Opcode::default_offset() + 8,
            3,
            U256Opcode::default_offset(),
        ),
        ExecutorName::BranchEqualRv32 => (
            BranchEqualOpcode::default_offset(),
            BranchEqualOpcode::COUNT,
            BranchEqualOpcode::default_offset(),
        ),
        ExecutorName::BranchLessThanRv32 => (
            BranchLessThanOpcode::default_offset(),
            BranchLessThanOpcode::COUNT,
            BranchLessThanOpcode::default_offset(),
        ),
        ExecutorName::Ui => (
            U32Opcode::default_offset(),
            U32Opcode::COUNT,
            U32Opcode::default_offset(),
        ),
        ExecutorName::CastF => (
            CastfOpcode::default_offset(),
            CastfOpcode::COUNT,
            CastfOpcode::default_offset(),
        ),
        ExecutorName::Secp256k1AddUnequal => {
            (EccOpcode::default_offset(), 1, EccOpcode::default_offset())
        }
        ExecutorName::Secp256k1Double => (
            EccOpcode::default_offset() + 1,
            1,
            EccOpcode::default_offset(),
        ),
    };
    (start..(start + len), offset)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    // Each executor handles the given range of opcode as usize (absolute, with offset).
    // Offset is the start opcode (usize) of the Opcode class, and it's needed because some Opcode classes are handled by different executors.
    // For example, U256Opcode class has some opcodes handled by ArithmeticLogicUnit256, and some by U256Multiplication.
    // And for U256Multiplication executor to verify the opcode it gets from program, it needs to know the offset of the U256Opcode class.
    pub executors: Vec<(Range<usize>, ExecutorName, usize)>, // (range of opcodes, who executes, offset)
    pub modular_executors: Vec<(Range<usize>, ExecutorName, usize, BigUint)>, // (range of opcodes, who executes, offset, modulus)

    pub poseidon2_max_constraint_degree: usize,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
    pub bigint_limb_size: usize,
}

impl VmConfig {
    pub fn from_parameters(
        poseidon2_max_constraint_degree: usize,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        bigint_limb_size: usize,
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
            bigint_limb_size,
            modular_executors: Vec::new(),
        };
        config.add_modular_support(enabled_modulus)
    }

    pub fn add_executor(
        mut self,
        range: Range<usize>,
        executor: ExecutorName,
        offset: usize,
    ) -> Self {
        // Some executors need to be handled in a special way, and cannot be added like other executors.
        let not_allowed_executors = [ExecutorName::ModularAddSub, ExecutorName::ModularMultDiv];
        if not_allowed_executors.contains(&executor) {
            panic!("Cannot add executor for {:?}", executor);
        }
        self.executors.push((range, executor, offset));
        self
    }

    pub fn add_default_executor(self, executor: ExecutorName) -> Self {
        let (range, offset) = default_executor_range(executor);
        self.add_executor(range, executor, offset)
    }

    // I think adding "opcode class" support is better than adding "executor".
    // The api should be saying: I want to be able to do this set of operations, and doesn't care about what executor is doing it.
    pub fn add_modular_support(self, enabled_modulus: Vec<BigUint>) -> Self {
        let mut res = self;
        let num_ops_per_modulus = ModularArithmeticOpcode::COUNT;
        for (i, modulus) in enabled_modulus.iter().enumerate() {
            let shift = i * num_ops_per_modulus;
            res = res.add_modular_prime(modulus, shift);
        }
        res
    }

    pub fn add_canonical_modulus(self) -> Self {
        let primes = Modulus::all().iter().map(|m| m.prime()).collect();
        self.add_modular_support(primes)
    }

    pub fn add_modular_prime(self, prime: &BigUint, shift: usize) -> Self {
        let add_sub_range = default_executor_range(ExecutorName::ModularAddSub);
        let mult_div_range = default_executor_range(ExecutorName::ModularMultDiv);
        let mut res = self;
        res.modular_executors.push((
            shift_range(&add_sub_range.0, shift),
            ExecutorName::ModularAddSub,
            add_sub_range.1 + shift,
            prime.clone(),
        ));
        res.modular_executors.push((
            shift_range(&mult_div_range.0, shift),
            ExecutorName::ModularMultDiv,
            mult_div_range.1 + shift,
            prime.clone(),
        ));
        res
    }

    pub fn add_ecc_support(self) -> Self {
        todo!()
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
            8,
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
            8,
            vec![],
        )
        .add_default_executor(ExecutorName::Core)
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
        Self::iter().collect()
    }
}

fn shift_range(r: &Range<usize>, x: usize) -> Range<usize> {
    let start = r.start + x;
    let end = r.end + x;
    start..end
}
