use std::sync::Arc;

pub use afs_primitives::bigint::utils::*;
use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerChip,
};
use air::ModularAddSubAir;
use hex_literal::hex;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use p3_field::PrimeField32;

use crate::{
    arch::{
        instructions::{Opcode, MODULAR_ADDSUB_INSTRUCTIONS},
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    memory::{MemoryChipRef, MemoryHeapReadRecord, MemoryHeapWriteRecord},
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
pub struct ModularAddSubRecord<T, const NUM_LIMBS: usize> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_array_read: MemoryHeapReadRecord<T, NUM_LIMBS>,
    pub y_array_read: MemoryHeapReadRecord<T, NUM_LIMBS>,
    pub z_array_write: MemoryHeapWriteRecord<T, NUM_LIMBS>,
}

// This chip is for modular addition and subtraction of usually 256 bit numbers
// represented as 32 8 bit limbs in little endian format.
// Warning: The chip can break if NUM_LIMBS * LIMB_SIZE is not equal to the number of bits in the modulus.
#[derive(Debug, Clone)]
pub struct ModularAddSubChip<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize> {
    pub air: ModularAddSubAir<NUM_LIMBS, LIMB_SIZE>,
    data: Vec<ModularAddSubRecord<T, NUM_LIMBS>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    modulus: BigUint,
}

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    ModularAddSubChip<T, NUM_LIMBS, LIMB_SIZE>
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
            air: ModularAddSubAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
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
    for ModularAddSubChip<T, NUM_LIMBS, LIMB_SIZE>
{
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
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
        assert!(MODULAR_ADDSUB_INSTRUCTIONS.contains(&opcode));
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

        self.data.push(ModularAddSubRecord {
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

impl<T: PrimeField32, const NUM_LIMBS: usize, const LIMB_SIZE: usize>
    ModularAddSubChip<T, NUM_LIMBS, LIMB_SIZE>
{
    pub fn solve(opcode: Opcode, mut x: BigUint, y: BigUint) -> BigUint {
        match opcode {
            Opcode::SECP256K1_COORD_ADD => (x + y) % SECP256K1_COORD_PRIME.clone(),
            Opcode::SECP256K1_SCALAR_ADD => (x + y) % SECP256K1_SCALAR_PRIME.clone(),
            Opcode::SECP256K1_COORD_SUB => {
                let tmp = SECP256K1_COORD_PRIME.clone();
                while x < y {
                    x += &tmp;
                }
                (x - y) % &tmp
            }
            Opcode::SECP256K1_SCALAR_SUB => {
                let tmp = SECP256K1_SCALAR_PRIME.clone();
                while x < y {
                    x += &tmp;
                }
                (x - y) % &tmp
            }
            _ => unreachable!(),
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
