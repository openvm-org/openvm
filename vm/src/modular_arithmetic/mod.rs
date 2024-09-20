use std::sync::Arc;

pub use afs_primitives::bigint::utils::*;
use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use air::ModularArithmeticAir;
use hex_literal::hex;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, MODULAR_ARITHMETIC_INSTRUCTIONS},
    },
    memory::{
        offline_checker::MemoryBridge, MemoryChipRef, MemoryHeapReadRecord, MemoryHeapWriteRecord,
    },
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

mod air;
mod bridge;
mod columns;
mod trace;

pub use columns::*;

#[cfg(test)]
mod tests;

// Max bits that can fit into our field element.
pub const FIELD_ELEMENT_BITS: usize = 30;

pub static SECP256K1_COORD_PRIME: Lazy<BigUint> = Lazy::new(|| {
    BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
    ))
});

pub static SECP256K1_SCALAR_PRIME: Lazy<BigUint> = Lazy::new(|| {
    BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
    ))
});

#[derive(Debug, Clone)]
pub struct ModularArithmeticRecord<T, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_array_read: MemoryHeapReadRecord<T, NUM_LIMBS>,
    pub y_array_read: MemoryHeapReadRecord<T, NUM_LIMBS>,
    pub z_array_write: MemoryHeapWriteRecord<T, NUM_LIMBS>,
}

#[derive(Clone, Debug)]
pub enum ModularArithmeticAirVariant {
    Add(ModularAdditionAir),
    Sub(ModularSubtractionAir),
    Mul(ModularMultiplicationAir),
    Div(ModularDivisionAir),
}

type TraceInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);
impl ModularArithmeticAirVariant {
    pub fn generate_trace_row<F: PrimeField64>(
        &self,
        input: TraceInput,
    ) -> ModularArithmeticCols<F> {
        match self {
            Self::Add(air) => LocalTraceInstructions::generate_trace_row(air, input),
            Self::Sub(air) => LocalTraceInstructions::generate_trace_row(air, input),
            Self::Mul(air) => LocalTraceInstructions::generate_trace_row(air, input),
            Self::Div(air) => LocalTraceInstructions::generate_trace_row(air, input),
        }
    }

    pub fn eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: ModularArithmeticCols<AB::Var>,
        aux: (),
    ) {
        match self {
            Self::Add(air) => SubAir::eval(air, builder, io, aux),
            Self::Sub(air) => SubAir::eval(air, builder, io, aux),
            Self::Mul(air) => SubAir::eval(air, builder, io, aux),
            Self::Div(air) => SubAir::eval(air, builder, io, aux),
        }
    }

    pub fn is_expected_opcode(&self, opcode: Opcode) -> bool {
        match self {
            Self::Add(_) => {
                [Opcode::SECP256K1_COORD_ADD, Opcode::SECP256K1_SCALAR_ADD].contains(&opcode)
            }
            Self::Sub(_) => {
                [Opcode::SECP256K1_COORD_SUB, Opcode::SECP256K1_SCALAR_SUB].contains(&opcode)
            }
            Self::Mul(_) => {
                [Opcode::SECP256K1_COORD_MUL, Opcode::SECP256K1_SCALAR_MUL].contains(&opcode)
            }
            Self::Div(_) => {
                [Opcode::SECP256K1_COORD_DIV, Opcode::SECP256K1_SCALAR_DIV].contains(&opcode)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum ModularArithmeticOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
pub struct ModularArithmeticVmAir<A> {
    pub air: A,
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
    pub memory_bridge: MemoryBridge,

    pub carry_limbs: usize,
    pub q_limbs: usize,
}

#[derive(Clone, Debug)]
pub struct ModularArithmeticChip<T: PrimeField32, A> {
    pub air: ModularArithmeticVmAir<A>,
    data: Vec<ModularArithmeticRecord<T>>,

    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    modulus: BigUint,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    ModularArithmeticChip<T, NUM_LIMBS, LIMB_SIZE>
{
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
        modulus: BigUint,
    ) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let memory_bridge = memory_chip.borrow().memory_bridge();
        let bus = range_checker_chip.bus();
        assert!(
            bus.range_max_bits >= LIMB_SIZE,
            "range_max_bits {} < LIMB_SIZE {}",
            bus.range_max_bits,
            LIMB_SIZE
        );
        let subair = CheckCarryModToZeroSubAir::new(
            modulus.clone(),
            LIMB_SIZE,
            bus.index,
            bus.range_max_bits,
            FIELD_ELEMENT_BITS,
        );
        Self {
            air: ModularArithmeticAir {
                execution_bus,
                program_bus,
                memory_bridge,
                subair,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
            modulus,
        }
    }
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize> InstructionExecutor<T>
    for ModularArithmeticChip<T, NUM_LIMBS, LIMB_SIZE>
{
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let mut memory_chip = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let Instruction {
            opcode,
            op_a: z_address_ptr,
            op_b: x_address_ptr,
            op_c: y_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        assert!(LIMB_SIZE <= 10); // refer to [primitives/src/bigint/README.md]
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&opcode));
        match opcode {
            Opcode::SECP256K1_COORD_ADD | Opcode::SECP256K1_COORD_SUB => {
                assert_eq!(self.modulus, SECP256K1_COORD_PRIME.clone());
            }
            Opcode::SECP256K1_SCALAR_ADD | Opcode::SECP256K1_SCALAR_SUB => {
                assert_eq!(self.modulus, SECP256K1_SCALAR_PRIME.clone());
            }
            _ => unreachable!(),
        }

        let mut memory_chip = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let x_array_read = memory_chip.read_heap::<NUM_LIMBS>(d, e, x_address_ptr);
        let y_array_read = memory_chip.read_heap::<NUM_LIMBS>(d, e, y_address_ptr);

        let x = x_array_read.data_read.data.map(|x| x.as_canonical_u32());
        let y = y_array_read.data_read.data.map(|x| x.as_canonical_u32());

        let x_biguint = Self::limbs_to_biguint(&x);
        let y_biguint = Self::limbs_to_biguint(&y);

        let z_biguint = Self::solve(opcode, x_biguint, y_biguint);
        let z_limbs = Self::biguint_to_limbs(z_biguint);

        let z_array_write = memory_chip.write_heap::<NUM_LIMBS>(
            d,
            e,
            z_address_ptr,
            z_limbs.map(|x| T::from_canonical_u32(x)),
        );

        self.data.push(ModularArithmeticRecord {
            from_state,
            instruction,
            x_array_read,
            y_array_read,
            z_array_write,
        });

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}

    // little endian.
    pub fn limbs_to_biguint(x: &[u32]) -> BigUint {
        let mut result = BigUint::zero();
        let base = BigUint::from_u32(1 << LIMB_SIZE).unwrap();
        for limb in x.iter().rev() {
            result = result * &base + BigUint::from_u32(*limb).unwrap();
        }
        result
    }

    // little endian.
    // Warning: This function only returns the last NUM_LIMBS*LIMB_SIZE bits of
    //          the input, while the input can have more than that.
    pub fn biguint_to_limbs(mut x: BigUint) -> [u32; NUM_LIMBS] {
        let mut result = [0; NUM_LIMBS];
        let base = BigUint::from_u32(1 << LIMB_SIZE).unwrap();
        for r in result.iter_mut() {
            *r = (x.clone() % &base).to_u32().unwrap();
            x /= &base;
        }
        result
    }
}
