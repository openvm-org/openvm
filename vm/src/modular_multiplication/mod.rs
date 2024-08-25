use std::{borrow::Cow, collections::VecDeque};

use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;
use num_bigint_dig::{algorithms::mod_inverse, BigUint};
use p3_field::{PrimeField32, PrimeField64};

use self::air::ModularArithmeticVmAir;
use crate::{
    cpu::{trace::Instruction, OpCode::*},
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

pub struct VmModularArithmetic<F: PrimeField64> {
    pub instruction: Instruction<F>,
    pub argument_1: BigUint,
    pub argument_2: BigUint,
}

pub struct ModularArithmeticChip<F: PrimeField64> {
    air: ModularArithmeticVmAir,
    ops: Vec<VmModularArithmetic<F>>,
}

impl<F: PrimeField32> ModularArithmeticChip<F> {
    pub fn new(air: ModularArithmeticVmAir) -> Self {
        Self { air, ops: vec![] }
    }

    pub fn calculate<const NUM_WORDS: usize, const WORD_SIZE: usize>(
        vm: &mut ExecutionSegment<NUM_WORDS, WORD_SIZE, F>,
        instruction: Instruction<F>,
    ) {
        let (op_input_2, op_result) = match instruction.opcode {
            SECP256K1_COORD_ADD | SECP256K1_COORD_MUL | SECP256K1_SCALAR_ADD
            | SECP256K1_SCALAR_MUL => (instruction.op_b, instruction.op_c),
            SECP256K1_COORD_SUB | SECP256K1_COORD_DIV | SECP256K1_SCALAR_SUB
            | SECP256K1_SCALAR_DIV => (instruction.op_c, instruction.op_b),
            _ => panic!(),
        };
        let modulus = match instruction.opcode {
            SECP256K1_COORD_ADD | SECP256K1_COORD_SUB | SECP256K1_COORD_MUL
            | SECP256K1_COORD_DIV => ModularArithmeticBigIntAir::secp256k1_coord_prime(),
            SECP256K1_SCALAR_ADD | SECP256K1_SCALAR_SUB | SECP256K1_SCALAR_MUL
            | SECP256K1_SCALAR_DIV => ModularArithmeticBigIntAir::secp256k1_scalar_prime(),
            _ => panic!(),
        };
        // TODO[zach]: update for word size
        let address1 = vm
            .memory_manager
            .borrow_mut()
            .read_word(instruction.d, instruction.op_a)
            .op
            .cell
            .data[0];
        let address2 = vm
            .memory_manager
            .borrow_mut()
            .read_word(instruction.d, op_input_2)
            .op
            .cell
            .data[0];
        let output_address = vm
            .memory_manager
            .borrow_mut()
            .read_word(instruction.d, op_result)
            .op
            .cell
            .data[0];

        let chip = vm.modular_arithmetic_chips.get_mut(&modulus).unwrap();
        let air = &chip.air;
        let num_elems = air.air.limb_dimensions.io_limb_sizes.len();
        let repr_bits = air.air.repr_bits;
        let argument_1_elems = (0..num_elems)
            .map(|i| {
                vm.memory_manager
                    .borrow_mut()
                    .read_word(instruction.e, address1 + F::from_canonical_usize(i))
                    .op
                    .cell
                    .data[0]
            })
            .collect();
        let argument_2_elems = (0..num_elems)
            .map(|i| {
                vm.memory_manager
                    .borrow_mut()
                    .read_word(instruction.e, address2 + F::from_canonical_usize(i))
                    .op
                    .cell
                    .data[0]
            })
            .collect();
        let argument_1 = elems_to_bigint(argument_1_elems, repr_bits);
        let argument_2 = elems_to_bigint(argument_2_elems, repr_bits);
        let result = match instruction.opcode {
            SECP256K1_COORD_ADD | SECP256K1_SCALAR_ADD => argument_1.clone() + argument_2.clone(),
            SECP256K1_COORD_SUB | SECP256K1_SCALAR_SUB => {
                argument_1.clone() + modulus.clone() - argument_2.clone()
            }
            SECP256K1_COORD_MUL | SECP256K1_SCALAR_MUL => argument_1.clone() * argument_2.clone(),
            SECP256K1_COORD_DIV | SECP256K1_SCALAR_DIV => {
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
            let mut word = [F::zero(); WORD_SIZE];
            word[0] = elem;

            vm.memory_manager.borrow_mut().write_word(
                instruction.e,
                output_address + F::from_canonical_usize(i),
                word,
            );
        }
        chip.ops.push(VmModularArithmetic {
            instruction,
            argument_1,
            argument_2,
        });
    }
}
