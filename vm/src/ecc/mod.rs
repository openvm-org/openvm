use std::sync::Arc;

use afs_primitives::{
    bigint::utils::big_uint_mod_inverse,
    ecc::{EcAddUnequalAir, EcAirConfig},
    var_range::VariableRangeCheckerChip,
};
use num_bigint_dig::BigUint;
use num_traits::FromPrimitive;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        // instructions::Opcode,
    },
    cpu::trace::Instruction,
    memory::{MemoryChipRef, MemoryHeapReadRecord, MemoryHeapWriteRecord},
    modular_arithmetic::{
        biguint_to_limbs, limbs_to_biguint, FIELD_ELEMENT_BITS, LIMB_SIZE, NUM_LIMBS,
        SECP256K1_COORD_PRIME, TWO_NUM_LIMBS,
    },
};

mod air;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;

#[cfg(test)]
mod test;

#[derive(Clone, Debug)]
pub struct EcAddUnequalRecord<T: PrimeField32> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    // Each limb is 8 bits (byte), 32 limbs for 256 bits, 2 coordinates for each point..
    pub p1_array_read: MemoryHeapReadRecord<T, TWO_NUM_LIMBS>,
    pub p2_array_read: MemoryHeapReadRecord<T, TWO_NUM_LIMBS>,
    pub p3_array_write: MemoryHeapWriteRecord<T, TWO_NUM_LIMBS>,
}

pub struct EcAddUnequalChip<T: PrimeField32> {
    pub air: EcAddUnequalVmAir,
    pub data: Vec<EcAddUnequalRecord<T>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
    prime: BigUint,
}

impl<T: PrimeField32> EcAddUnequalChip<T> {
    pub fn new(execution_bus: ExecutionBus, memory_chip: MemoryChipRef<T>) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let memory_bridge = memory_chip.borrow().memory_bridge();
        let prime = SECP256K1_COORD_PRIME.clone();

        let ec_config = EcAirConfig::new(
            prime.clone(),
            BigUint::from_u32(7).unwrap(),
            range_checker_chip.bus().index,
            range_checker_chip.range_max_bits(),
            LIMB_SIZE,
            FIELD_ELEMENT_BITS,
        );
        let air = EcAddUnequalVmAir {
            air: EcAddUnequalAir { config: ec_config },
            execution_bus,
            memory_bridge,
        };

        Self {
            air,
            data: Vec::new(),
            memory_chip,
            range_checker_chip,
            prime,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for EcAddUnequalChip<T> {
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
            opcode: _,
            op_a: p3_address_ptr,
            op_b: p1_address_ptr,
            op_c: p2_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();

        // TODO: check opcode

        let p1_array_read = memory_chip.read_heap::<TWO_NUM_LIMBS>(d, e, p1_address_ptr);
        let p2_array_read = memory_chip.read_heap::<TWO_NUM_LIMBS>(d, e, p2_address_ptr);
        let p1_array = p1_array_read.data_read.data.map(|x| x.as_canonical_u32());
        let p1_x = limbs_to_biguint(&p1_array[..NUM_LIMBS]);
        let p1_y = limbs_to_biguint(&p1_array[NUM_LIMBS..]);
        let p2_array = p2_array_read.data_read.data.map(|x| x.as_canonical_u32());
        let p2_x = limbs_to_biguint(&p2_array[..NUM_LIMBS]);
        let p2_y = limbs_to_biguint(&p2_array[NUM_LIMBS..]);

        let dx = &self.prime + &p1_x - &p2_x;
        let dy = &self.prime + &p1_y - &p2_y;
        let dx_inv = big_uint_mod_inverse(&dx, &self.prime);
        let lambda: BigUint = (dy * dx_inv) % &self.prime;
        let p3_x: BigUint =
            (&lambda * &lambda + &self.prime + &self.prime - &p1_x - &p2_x) % &self.prime;
        let p3_y: BigUint =
            (&lambda * (&self.prime + &p1_x - &p3_x) + &self.prime - &p1_y) % &self.prime;

        let p3_x_limbs = biguint_to_limbs(p3_x);
        let p3_y_limbs = biguint_to_limbs(p3_y);
        let mut p3_array = [0; 64];
        p3_array[..NUM_LIMBS].copy_from_slice(&p3_x_limbs);
        p3_array[NUM_LIMBS..].copy_from_slice(&p3_y_limbs);
        let p3_array: [T; 64] = p3_array.map(|x| T::from_canonical_u32(x));
        let p3_array_write =
            memory_chip.write_heap::<TWO_NUM_LIMBS>(d, e, p3_address_ptr, p3_array);

        let record = EcAddUnequalRecord {
            from_state,
            instruction,
            p1_array_read,
            p2_array_read,
            p3_array_write,
        };
        self.data.push(record);

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        }
    }
}
