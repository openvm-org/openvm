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

#[derive(Debug, Serialize, Deserialize, Clone, Copy, new)]
pub struct MemoryConfig {
    pub addr_space_max_bits: usize,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    pub decomp: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new(29, 29, 29, 16)
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
        ExecutorName::ArithmeticLogicUnit256 => (
            U256Opcode::default_offset(),
            8,
            U256Opcode::default_offset(),
        ),
        ExecutorName::U256Multiplication => (
            U256Opcode::default_offset() + 11,
            1,
            U256Opcode::default_offset(),
        ),
        ExecutorName::Shift256 => (
            U256Opcode::default_offset() + 8,
            3,
            U256Opcode::default_offset(),
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
    // Each executor handles the given range of opcode as usize, in absolute sense.
    // offset is needed as for a given group of opcodes, they are actually mapped to different executors.
    // For example, U256Opcode class has some opcodes handled by ArithmeticLogicUnit256, and some by U256Multiplication.
    // And for U256Multiplication executor to verify the opcode it gets from program, it needs to know the offset of the U256Opcode class.
    pub executors: Vec<(Range<usize>, ExecutorName, usize)>, // (range of opcodes, who executes, offset)
    pub modular_executors: Vec<(Range<usize>, ExecutorName, usize, BigUint)>, // (range of opcodes, who executes, offset, modulus)

    pub poseidon2_max_constraint_degree: Option<usize>,
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
        poseidon2_max_constraint_degree: Option<usize>,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        bigint_limb_size: usize,
    ) -> Self {
        VmConfig {
            executors: Vec::new(),
            poseidon2_max_constraint_degree,
            memory_config,
            num_public_values,
            max_segment_len,
            collect_metrics,
            bigint_limb_size,
            modular_executors: Vec::new(),
        }
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
    pub fn add_modular_support(self) -> Self {
        let add_sub_range = default_executor_range(ExecutorName::ModularAddSub);
        let mult_div_range = default_executor_range(ExecutorName::ModularMultDiv);
        let num_ops_per_modulus = ModularArithmeticOpcode::COUNT;
        let mut res = self;
        for modulus in Modulus::iter() {
            let modulus_usize = modulus.clone() as usize;
            let shift = modulus_usize * num_ops_per_modulus;
            res.modular_executors.push((
                shift_range(&add_sub_range.0, shift),
                ExecutorName::ModularAddSub,
                add_sub_range.1 + shift,
                modulus.prime(),
            ));
            res.modular_executors.push((
                shift_range(&mult_div_range.0, shift),
                ExecutorName::ModularMultDiv,
                mult_div_range.1 + shift,
                modulus.prime(),
            ));
        }
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
            Some(DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE),
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            8,
        )
    }

    pub fn core_options(&self) -> CoreOptions {
        CoreOptions {
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        Self::from_parameters(
            None,
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            8,
        )
        .add_default_executor(ExecutorName::Core)
    }

    pub fn aggregation(poseidon2_max_constraint_degree: usize) -> Self {
        VmConfig {
            poseidon2_max_constraint_degree: Some(poseidon2_max_constraint_degree),
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
}

fn shift_range(r: &Range<usize>, x: usize) -> Range<usize> {
    let start = r.start + x;
    let end = r.end + x;
    start..end
}
