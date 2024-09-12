use std::sync::Arc;

use afs_primitives::{
    bigint::{
        modular_arithmetic::{
            add::ModularAdditionAir, div::ModularDivisionAir, mul::ModularMultiplicationAir,
            sub::ModularSubtractionAir, ModularArithmeticCols,
        },
        utils::{big_uint_mod_inverse, get_arithmetic_air},
    },
    sub_chip::{LocalTraceInstructions, SubAir},
    var_range::VariableRangeCheckerChip,
};
use afs_stark_backend::interaction::InteractionBuilder;
use hex_literal::hex;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use p3_field::{PrimeField32, PrimeField64};

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
pub enum PrimitiveArithmeticAir {
    Add(ModularAdditionAir),
    Sub(ModularSubtractionAir),
    Mul(ModularMultiplicationAir),
    Div(ModularDivisionAir),
}

type TraceInput = (BigUint, BigUint, Arc<VariableRangeCheckerChip>);
impl PrimitiveArithmeticAir {
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
pub struct ModularArithmeticAir<A> {
    pub air: A,
    pub execution_bus: ExecutionBus,
    pub mem_oc: MemoryOfflineChecker,

    pub carry_limbs: usize,
    pub q_limbs: usize,
}

#[derive(Clone, Debug)]
pub struct ModularArithmeticChip<T: PrimeField32, A> {
    pub air: ModularArithmeticAir<A>,
    data: Vec<ModularArithmeticRecord<T>>,

    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,

    modulus: BigUint,
}

impl<T: PrimeField32> ModularArithmeticChip<T, PrimitiveArithmeticAir> {
    pub fn new(
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<T>,
        modulus: BigUint,
        op: ModularArithmeticOp,
    ) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let mem_oc = memory_chip.borrow().make_offline_checker();
        let (carry_limbs, q_limbs, is_mul_div) = match op {
            ModularArithmeticOp::Add | ModularArithmeticOp::Sub => (NUM_LIMBS, 1, false),
            ModularArithmeticOp::Mul | ModularArithmeticOp::Div => {
                (NUM_LIMBS * 2 - 1, NUM_LIMBS, true)
            }
        };
        let primitive_arithmetic_air = get_arithmetic_air(
            modulus.clone(),
            LIMB_SIZE,
            FIELD_ELEMENT_BITS,
            NUM_LIMBS,
            is_mul_div,
            range_checker_chip.bus().index,
            range_checker_chip.bus().range_max_bits,
        );
        let subair = match op {
            ModularArithmeticOp::Add => PrimitiveArithmeticAir::Add(ModularAdditionAir {
                arithmetic: primitive_arithmetic_air,
            }),
            ModularArithmeticOp::Sub => PrimitiveArithmeticAir::Sub(ModularSubtractionAir {
                arithmetic: primitive_arithmetic_air,
            }),
            ModularArithmeticOp::Mul => PrimitiveArithmeticAir::Mul(ModularMultiplicationAir {
                arithmetic: primitive_arithmetic_air,
            }),
            ModularArithmeticOp::Div => PrimitiveArithmeticAir::Div(ModularDivisionAir {
                arithmetic: primitive_arithmetic_air,
            }),
        };

        Self {
            air: ModularArithmeticAir {
                air: subair,
                execution_bus,
                mem_oc,
                carry_limbs,
                q_limbs,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
            modulus,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for ModularArithmeticChip<T, PrimitiveArithmeticAir> {
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
            op_a: z_address_ptr,
            op_b: x_address_ptr,
            op_c: y_address_ptr,
            d,
            e,
            ..
        } = instruction.clone();
        assert!(self.air.air.is_expected_opcode(opcode));

        // let (y_address_ptr, z_address_ptr) = match opcode {
        //     Opcode::SECP256K1_COORD_ADD
        //     | Opcode::SECP256K1_SCALAR_ADD
        //     | Opcode::SECP256K1_COORD_MUL
        //     | Opcode::SECP256K1_SCALAR_MUL => (b_address_ptr, c_address_ptr),
        //     Opcode::SECP256K1_COORD_SUB
        //     | Opcode::SECP256K1_SCALAR_SUB
        //     | Opcode::SECP256K1_COORD_DIV
        //     | Opcode::SECP256K1_SCALAR_DIV => (c_address_ptr, b_address_ptr),
        //     _ => panic!(),
        // };

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
pub fn biguint_to_limbs(mut x: BigUint) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let base = BigUint::from_u32(1 << LIMB_SIZE).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result
}
