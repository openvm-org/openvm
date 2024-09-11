use std::{marker::PhantomData, sync::Arc};

use afs_primitives::var_range::VariableRangeCheckerChip;
use air::UintArithmeticAir;
use itertools::Itertools;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, UINT256_ARITHMETIC_INSTRUCTIONS},
    },
    cpu::trace::Instruction,
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const NUM_LIMBS: usize = 32; // This is used in some places where const generics are hard to use.
                                 // Of course, TODO make it something normal

pub const fn num_limbs<const ARG_SIZE: usize, const LIMB_SIZE: usize>() -> usize {
    (ARG_SIZE + LIMB_SIZE - 1) / LIMB_SIZE
}

#[derive(Debug)]
pub enum WriteRecord<T> {
    Uint(MemoryWriteRecord<NUM_LIMBS, T>),
    Short(MemoryWriteRecord<1, T>),
}

#[derive(Debug)]
pub struct UintArithmeticRecord<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_ptr_read: MemoryReadRecord<1, T>,
    pub y_ptr_read: MemoryReadRecord<1, T>,
    pub z_ptr_read: MemoryReadRecord<1, T>,

    pub x_read: MemoryReadRecord<NUM_LIMBS, T>,
    pub y_read: MemoryReadRecord<NUM_LIMBS, T>,
    pub z_write: WriteRecord<T>,

    // this may be redundant because we can extract it from z_write,
    // but it's not always the case
    pub result: Vec<T>,

    pub buffer: Vec<T>,
}

#[derive(Debug)]
pub struct UintArithmeticChip<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32> {
    pub air: UintArithmeticAir<ARG_SIZE, LIMB_SIZE>,
    data: Vec<UintArithmeticRecord<ARG_SIZE, LIMB_SIZE, T>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32>
    UintArithmeticChip<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn new(execution_bus: ExecutionBus, memory_chip: MemoryChipRef<T>) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let bus = range_checker_chip.bus();
        let mem_oc = memory_chip.borrow().make_offline_checker();
        assert!(
            bus.range_max_bits >= LIMB_SIZE,
            "range_max_bits {} < LIMB_SIZE {}",
            bus.range_max_bits,
            LIMB_SIZE
        );
        Self {
            air: UintArithmeticAir {
                execution_bus,
                mem_oc,
                bus,
                base_op: Opcode::ADD256,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
        }
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32> InstructionExecutor<T>
    for UintArithmeticChip<ARG_SIZE, LIMB_SIZE, T>
{
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            op_c: c,
            d,
            e,
            op_f: f,
            op_g: g,
            ..
        } = instruction.clone();
        assert!(UINT256_ARITHMETIC_INSTRUCTIONS.contains(&opcode));

        let mut memory_chip = self.memory_chip.borrow_mut();

        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let [z_ptr_read, x_ptr_read, y_ptr_read] =
            [a, b, c].map(|ptr_of_ptr| memory_chip.read_cell(d, ptr_of_ptr));

        let x_read = memory_chip.read::<NUM_LIMBS>(f, x_ptr_read.value());
        let y_read = memory_chip.read::<NUM_LIMBS>(g, y_ptr_read.value());

        let x = x_read.data.map(|x| x.as_canonical_u32());
        let y = y_read.data.map(|x| x.as_canonical_u32());
        let (z, residue) = UintArithmetic::<ARG_SIZE, LIMB_SIZE, T>::solve(opcode, (&x, &y));
        let CalculationResidue { result, buffer } = residue;

        let z_address_space = e;
        let z_write: WriteRecord<T> = match z {
            CalculationResult::Uint(limbs) => {
                let to_write = limbs
                    .iter()
                    .map(|x| T::from_canonical_u32(*x))
                    .collect::<Vec<_>>();
                WriteRecord::Uint(memory_chip.write::<NUM_LIMBS>(
                    z_address_space,
                    z_ptr_read.value(),
                    to_write.try_into().unwrap(),
                ))
            }
            CalculationResult::Short(res) => {
                println!("writing cell: {:?} {:?}", e, z_ptr_read.value());
                WriteRecord::Short(memory_chip.write_cell(e, z_ptr_read.value(), T::from_bool(res)))
            }
        };

        for elem in result.iter() {
            self.range_checker_chip.add_count(*elem, LIMB_SIZE);
        }

        self.data.push(UintArithmeticRecord {
            from_state,
            instruction: instruction.clone(),
            x_ptr_read,
            y_ptr_read,
            z_ptr_read,
            x_read,
            y_read,
            z_write,
            result: result.into_iter().map(T::from_canonical_u32).collect_vec(),
            buffer: buffer.into_iter().map(T::from_canonical_u32).collect_vec(),
        });

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        }
    }
}

pub enum CalculationResult<T> {
    Uint(Vec<T>),
    Short(bool),
}

pub struct CalculationResidue<T> {
    pub result: Vec<T>,
    pub buffer: Vec<T>,
}

pub struct UintArithmetic<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32> {
    _marker: PhantomData<F>,
}
impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32>
    UintArithmetic<ARG_SIZE, LIMB_SIZE, F>
{
    pub fn solve(
        opcode: Opcode,
        (x, y): (&[u32], &[u32]),
    ) -> (CalculationResult<u32>, CalculationResidue<u32>) {
        match opcode {
            Opcode::ADD256 => {
                let (result, carry) = Self::add(x, y);
                (
                    CalculationResult::Uint(result.clone()),
                    CalculationResidue {
                        result,
                        buffer: carry,
                    },
                )
            }
            Opcode::SUB256 => {
                let (result, carry) = Self::subtract(x, y);
                (
                    CalculationResult::Uint(result.clone()),
                    CalculationResidue {
                        result,
                        buffer: carry,
                    },
                )
            }
            Opcode::LT256 => {
                let (diff, carry) = Self::subtract(x, y);
                let cmp_result = *carry.last().unwrap() == 1;
                (
                    CalculationResult::Short(cmp_result),
                    CalculationResidue {
                        result: diff,
                        buffer: carry,
                    },
                )
            }
            Opcode::EQ256 => {
                let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
                let mut inverse = vec![0u32; num_limbs];
                for i in 0..num_limbs {
                    if x[i] != y[i] {
                        inverse[i] = (F::from_canonical_u32(x[i]) - F::from_canonical_u32(y[i]))
                            .inverse()
                            .as_canonical_u32();
                        break;
                    }
                }
                (
                    CalculationResult::Short(x == y),
                    CalculationResidue {
                        result: Default::default(),
                        buffer: inverse,
                    },
                )
            }
            _ => unreachable!(),
        }
    }

    fn add(x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        let mut result = vec![0u32; num_limbs];
        let mut carry = vec![0u32; num_limbs];
        for i in 0..num_limbs {
            result[i] = x[i] + y[i] + if i > 0 { carry[i - 1] } else { 0 };
            carry[i] = result[i] >> LIMB_SIZE;
            result[i] &= (1 << LIMB_SIZE) - 1;
        }
        (result, carry)
    }

    fn subtract(x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();
        let mut result = vec![0u32; num_limbs];
        let mut carry = vec![0u32; num_limbs];
        for i in 0..num_limbs {
            let rhs = y[i] + if i > 0 { carry[i - 1] } else { 0 };
            if x[i] >= rhs {
                result[i] = x[i] - rhs;
                carry[i] = 0;
            } else {
                result[i] = x[i] + (1 << LIMB_SIZE) - rhs;
                carry[i] = 1;
            }
        }
        (result, carry)
    }
}
