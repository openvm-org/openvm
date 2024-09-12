use std::sync::Arc;

use afs_primitives::{
    bigint::{
        modular_arithmetic::add::ModularAdditionAir,
        utils::{big_uint_mod_inverse, get_arithmetic_air},
    },
    var_range::VariableRangeCheckerChip,
};
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode,
    },
    cpu::trace::Instruction,
    memory::{
        offline_checker::MemoryOfflineChecker, MemoryChipRef, MemoryReadRecord, MemoryWriteRecord,
    },
};

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[cfg(test)]
mod tests;

// Current assumption: modulus is 256 bits.
// We use 8-bits limb, and so 32 limbs for each operand.
// TODO: maybe use const generic.
pub const LIMB_SIZE: usize = 8;
pub const NUM_LIMBS: usize = 32;

// Max bits that can fit into our field element.
pub const FIELD_ELEMENT_BITS: usize = 30;

#[derive(Clone, Debug)]
pub struct ModularArithmeticRecord<T: PrimeField32> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_address_read: MemoryReadRecord<T, 1>,
    pub y_address_read: MemoryReadRecord<T, 1>,
    pub z_address_read: MemoryReadRecord<T, 1>,
    // Each limb is 8 bits (byte), 32 limbs for 256 bits.
    pub x_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub y_read: MemoryReadRecord<T, NUM_LIMBS>,
    pub z_write: MemoryWriteRecord<T, NUM_LIMBS>,
}

#[derive(Clone, Debug)]
pub struct ModularArithmeticAir {
    pub air: ModularAdditionAir,
    pub execution_bus: ExecutionBus,
    pub mem_oc: MemoryOfflineChecker,

    pub carry_limbs: usize,
    pub q_limbs: usize,
}

#[derive(Clone, Debug)]
pub struct ModularArithmeticChip<T: PrimeField32> {
    pub air: ModularArithmeticAir,
    data: Vec<ModularArithmeticRecord<T>>,

    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,

    modulus: BigUint,
}

impl<T: PrimeField32> ModularArithmeticChip<T> {
    pub fn new(
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<T>,
        modulus: BigUint,
    ) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let mem_oc = memory_chip.borrow().make_offline_checker();
        let primitive_arithmetic_addsub = get_arithmetic_air(
            modulus.clone(),
            LIMB_SIZE,
            FIELD_ELEMENT_BITS,
            NUM_LIMBS,
            false,
            range_checker_chip.bus().index,
            range_checker_chip.bus().range_max_bits,
        );
        let add_subair = ModularAdditionAir {
            arithmetic: primitive_arithmetic_addsub,
        };

        Self {
            air: ModularArithmeticAir {
                air: add_subair,
                execution_bus,
                mem_oc,
                // FIXME: it's different for mul/div
                carry_limbs: NUM_LIMBS,
                q_limbs: 1,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
            modulus,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for ModularArithmeticChip<T> {
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let mut memory_chip = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let Instruction {
            opcode,
            op_a: x_address_ptr,
            op_b: y_address_ptr,
            op_c: z_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();

        let x_address_read = memory_chip.read_cell(d, x_address_ptr);
        let y_address_read = memory_chip.read_cell(d, y_address_ptr);
        let z_address_read = memory_chip.read_cell(d, z_address_ptr);

        let x_read = memory_chip.read::<NUM_LIMBS>(e, x_address_read.value());
        let y_read = memory_chip.read::<NUM_LIMBS>(e, y_address_read.value());

        let x = x_read.data.map(|x| x.as_canonical_u32());
        let y = y_read.data.map(|x| x.as_canonical_u32());
        let mut x_biguint = limbs_to_biguint(&x);
        let y_biguint = limbs_to_biguint(&y);

        let z_biguint = match opcode {
            Opcode::SECP256K1_COORD_ADD | Opcode::SECP256K1_SCALAR_ADD => {
                (x_biguint + y_biguint) % &self.modulus
            }
            Opcode::SECP256K1_COORD_SUB | Opcode::SECP256K1_SCALAR_SUB => {
                while x_biguint < y_biguint {
                    x_biguint += &self.modulus;
                }
                (x_biguint - y_biguint) % &self.modulus
            }
            Opcode::SECP256K1_COORD_MUL | Opcode::SECP256K1_SCALAR_MUL => {
                (x_biguint * y_biguint) % &self.modulus
            }
            Opcode::SECP256K1_COORD_DIV | Opcode::SECP256K1_SCALAR_DIV => {
                let y_inv = big_uint_mod_inverse(&y_biguint, &self.modulus);

                (x_biguint * y_inv) % &self.modulus
            }
            _ => {
                println!("op {:?}", opcode);
                unreachable!()
            }
        };
        let z_limbs = biguint_to_limbs(z_biguint);

        let z_write = memory_chip.write::<NUM_LIMBS>(
            e,
            z_address_read.value(),
            z_limbs.map(|x| T::from_canonical_u32(x)),
        );

        let record = ModularArithmeticRecord {
            from_state,
            instruction,
            x_address_read,
            y_address_read,
            z_address_read,
            x_read,
            y_read,
            z_write,
        };
        self.data.push(record);

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        }
    }
}

fn limbs_to_biguint(x: &[u32]) -> BigUint {
    let mut result = BigUint::zero();
    let base = BigUint::from_u32(1 << LIMB_SIZE).unwrap();
    for limb in x {
        result = result * &base + BigUint::from_u32(*limb).unwrap();
    }
    result
}

fn biguint_to_limbs(mut x: BigUint) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let base = BigUint::from_u32(1 << LIMB_SIZE).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result.reverse();
    result
}
