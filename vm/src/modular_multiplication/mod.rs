use std::{borrow::Cow, collections::VecDeque};

use afs_primitives::modular_multiplication::bigint::air::ModularMultiplicationBigIntAir;
use num_bigint_dig::{algorithms::mod_inverse, BigUint};
use p3_field::{PrimeField32, PrimeField64};

use crate::{
    cpu::{trace::Instruction, OpCode::*},
    modular_multiplication::air::ModularMultiplicationVmAir,
    vm::ExecutionSegment,
};

pub mod air;
//mod columns;
#[cfg(test)]
mod tests;

pub fn elems_to_bigint<F: PrimeField64>(elems: Vec<F>, repr_bits: usize) -> BigUint {
    let mut bits = vec![];
    for elem in elems {
        let mut elem = elem.as_canonical_u64() as usize;
        for _ in 0..repr_bits {
            bits.push(elem & 1);
            elem /= 2;
        }
    }
    let mut bytes = vec![];
    for i in (0..bits.len()).step_by(8) {
        let mut byte = 0;
        for j in 0..8 {
            if i + j < bits.len() {
                byte += (bits[i + j] << j) as u8;
            }
        }
        bytes.push(byte)
    }
    BigUint::from_bytes_le(&bytes)
}

fn big_uint_to_bits(x: BigUint) -> VecDeque<usize> {
    let mut result = VecDeque::new();
    for byte in x.to_bytes_le() {
        for i in 0..8 {
            result.push_back(((byte >> i) as usize) & 1);
        }
    }
    result
}

fn take_limb(deque: &mut VecDeque<usize>, limb_size: usize) -> usize {
    if limb_size == 0 {
        0
    } else {
        let bit = deque.pop_front().unwrap_or(0);
        bit + (2 * take_limb(deque, limb_size - 1))
    }
}

pub fn bigint_to_elems<F: PrimeField64>(
    bigint: BigUint,
    repr_bits: usize,
    num_elems: usize,
) -> Vec<F> {
    let mut bits = big_uint_to_bits(bigint);
    (0..num_elems)
        .map(|_| F::from_canonical_usize(take_limb(&mut bits, repr_bits)))
        .collect()
}

pub struct VmModularMultiplication<F: PrimeField64> {
    pub instruction: Instruction<F>,
    pub argument_1: BigUint,
    pub argument_2: BigUint,
}

pub struct ModularMultiplicationChip<F: PrimeField64> {
    air: ModularMultiplicationVmAir,
    multiplications: Vec<VmModularMultiplication<F>>,
}

impl<F: PrimeField32> ModularMultiplicationChip<F> {
    pub fn new(air: ModularMultiplicationBigIntAir) -> Self {
        Self {
            air: ModularMultiplicationVmAir { air },
            multiplications: vec![],
        }
    }

    pub fn calculate<const WORD_SIZE: usize>(
        vm: &mut ExecutionSegment<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) {
        let mut timestamp = start_timestamp;
        let mut next_timestamp = || {
            timestamp += 1;
            timestamp - 1
        };
        let (op_input_2, op_result) = match instruction.opcode {
            MOD_SECP256K1_ADD | MOD_SECP256K1_MUL => (instruction.op_b, instruction.op_c),
            MOD_SECP256K1_SUB | MOD_SECP256K1_DIV => (instruction.op_c, instruction.op_b),
            _ => panic!(),
        };
        let address1 = vm
            .memory_chip
            .read_elem(next_timestamp(), instruction.d, instruction.op_a);
        let address2 = vm
            .memory_chip
            .read_elem(next_timestamp(), instruction.d, op_input_2);
        let output_address = vm
            .memory_chip
            .read_elem(next_timestamp(), instruction.d, op_result);

        let air = &vm.modular_multiplication_chip.air.air;
        let modulus = air.modulus.clone();
        let num_elems = air.limb_dimensions.io_limb_sizes.len();
        let repr_bits = air.repr_bits;
        let argument_1_elems = (0..num_elems)
            .map(|i| {
                vm.memory_chip.read_elem(
                    next_timestamp(),
                    instruction.e,
                    address1 + F::from_canonical_usize(i),
                )
            })
            .collect();
        let argument_2_elems = (0..num_elems)
            .map(|i| {
                vm.memory_chip.read_elem(
                    next_timestamp(),
                    instruction.e,
                    address2 + F::from_canonical_usize(i),
                )
            })
            .collect();
        let argument_1 = elems_to_bigint(argument_1_elems, repr_bits);
        let argument_2 = elems_to_bigint(argument_2_elems, repr_bits);
        let result = match instruction.opcode {
            MOD_SECP256K1_ADD => argument_1.clone() + argument_2.clone(),
            MOD_SECP256K1_SUB => argument_1.clone() + modulus.clone() - argument_2.clone(),
            MOD_SECP256K1_MUL => argument_1.clone() * argument_2.clone(),
            MOD_SECP256K1_DIV => {
                argument_1.clone()
                    * mod_inverse(Cow::Borrowed(&argument_2), Cow::Borrowed(&modulus))
                        .unwrap()
                        .to_biguint()
                        .unwrap()
            }
            _ => panic!(),
        } % modulus;
        let result_elems = bigint_to_elems(result, repr_bits, num_elems);
        for (i, &elem) in result_elems.iter().enumerate() {
            vm.memory_chip.write_elem(
                next_timestamp(),
                instruction.e,
                output_address + F::from_canonical_usize(i),
                elem,
            );
        }
        vm.modular_multiplication_chip
            .multiplications
            .push(VmModularMultiplication {
                instruction,
                argument_1,
                argument_2,
            });
    }
}
