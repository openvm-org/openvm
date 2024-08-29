use std::{marker::PhantomData, sync::Arc};

use afs_primitives::{range::bus::RangeCheckBus, range_gate::RangeCheckerGateChip};
use air::LongArithmeticAir;
use itertools::Itertools;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, LONG_ARITHMETIC_INSTRUCTIONS},
    },
    cpu::trace::Instruction,
    memory::manager::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const fn num_limbs<const ARG_SIZE: usize, const LIMB_SIZE: usize>() -> usize {
    (ARG_SIZE + LIMB_SIZE - 1) / LIMB_SIZE
}

pub enum WriteRecord<T> {
    Long(MemoryWriteRecord<16, T>),
    Short(MemoryWriteRecord<1, T>),
}

pub struct LongArithmeticRecord<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_read: MemoryReadRecord<16, T>, // TODO: 16 -> generic expr or smth
    pub y_read: MemoryReadRecord<16, T>, // TODO: 16 -> generic expr or smth
    pub z_write: WriteRecord<T>,
}

pub struct LongArithmeticExecutionData<const ARG_SIZE: usize, const LIMB_SIZE: usize, T> {
    pub record: LongArithmeticRecord<ARG_SIZE, LIMB_SIZE, T>,

    // this may be redundant because we can extract it from z_write,
    // but it's not always the case
    pub result: Vec<T>,

    pub buffer: Vec<T>,
}

pub struct LongArithmeticChip<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32> {
    pub air: LongArithmeticAir<ARG_SIZE, LIMB_SIZE>,
    data: Vec<LongArithmeticExecutionData<ARG_SIZE, LIMB_SIZE, T>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<RangeCheckerGateChip>,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32>
    LongArithmeticChip<ARG_SIZE, LIMB_SIZE, T>
{
    pub fn new(
        bus: RangeCheckBus,
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<T>,
    ) -> Self {
        let mem_oc = memory_chip.borrow().make_offline_checker();
        Self {
            air: LongArithmeticAir {
                execution_bus,
                mem_oc,
                bus,
                base_op: Opcode::ADD256,
            },
            data: vec![],
            memory_chip,
            range_checker_chip: RangeCheckerGateChip::new(bus).into(),
        }
    }
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: PrimeField32> InstructionExecutor<T>
    for LongArithmeticChip<ARG_SIZE, LIMB_SIZE, T>
{
    fn execute(
        &mut self,
        instruction: &Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: z_address,
            op_b: x_address,
            op_c: y_address,
            d: z_as,
            e: x_as,
            op_f: y_as,
            ..
        } = instruction.clone();
        assert!(LONG_ARITHMETIC_INSTRUCTIONS.contains(&opcode));

        let mut memory_chip = self.memory_chip.borrow_mut();

        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let x_read = memory_chip.read::<16>(x_as, x_address); // TODO: 16 -> generic expr or smth
        let y_read = memory_chip.read::<16>(y_as, y_address); // TODO: 16 -> generic expr or smth

        let x = x_read.data.map(|x| x.as_canonical_u32());
        let y = y_read.data.map(|x| x.as_canonical_u32());
        let (z, residue) = LongArithmetic::<ARG_SIZE, LIMB_SIZE, T>::solve(opcode, (&x, &y));
        let CalculationResidue { result, buffer } = residue;

        let z_write: WriteRecord<T> = match z {
            CalculationResult::Long(limbs) => {
                let to_write = limbs
                    .iter()
                    .map(|x| T::from_canonical_u32(*x))
                    .collect::<Vec<_>>();
                WriteRecord::Long(memory_chip.write::<16>(
                    z_as,
                    z_address,
                    to_write.try_into().unwrap(),
                ))
            }
            CalculationResult::Short(res) => {
                WriteRecord::Short(memory_chip.write_cell(z_as, z_address, T::from_bool(res)))
            }
        };

        for elem in result.iter() {
            self.range_checker_chip.add_count(*elem);
        }

        self.data.push(LongArithmeticExecutionData {
            record: LongArithmeticRecord {
                from_state,
                instruction: instruction.clone(),
                x_read,
                y_read,
                z_write,
            },
            result: result.into_iter().map(T::from_canonical_u32).collect_vec(),
            buffer: buffer.into_iter().map(T::from_canonical_u32).collect_vec(),
        });

        let timestamp_delta = num_limbs::<ARG_SIZE, LIMB_SIZE>() * 2
            + if opcode == Opcode::ADD256 || opcode == Opcode::SUB256 {
                num_limbs::<ARG_SIZE, LIMB_SIZE>()
            } else {
                1
            };
        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + timestamp_delta,
        }
    }
}

pub enum CalculationResult<T> {
    Long(Vec<T>),
    Short(bool),
}

pub struct CalculationResidue<T> {
    pub result: Vec<T>,
    pub buffer: Vec<T>,
}

pub struct LongArithmetic<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32> {
    _marker: PhantomData<F>,
}
impl<const ARG_SIZE: usize, const LIMB_SIZE: usize, F: PrimeField32>
    LongArithmetic<ARG_SIZE, LIMB_SIZE, F>
{
    pub fn solve(
        opcode: Opcode,
        (x, y): (&[u32], &[u32]),
    ) -> (CalculationResult<u32>, CalculationResidue<u32>) {
        match opcode {
            Opcode::ADD256 => {
                let (result, carry) = Self::calc_sum(x, y);
                (
                    CalculationResult::Long(result.clone()),
                    CalculationResidue {
                        result,
                        buffer: carry,
                    },
                )
            }
            Opcode::SUB256 => {
                let (result, carry) = Self::calc_diff(x, y);
                (
                    CalculationResult::Long(result.clone()),
                    CalculationResidue {
                        result,
                        buffer: carry,
                    },
                )
            }
            Opcode::LT256 => {
                let (diff, carry) = Self::calc_diff(x, y);
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
                    CalculationResult::Short(x.iter().zip(y).all(|(x, y)| x == y)),
                    CalculationResidue {
                        result: Default::default(),
                        buffer: inverse,
                    },
                )
            }
            _ => unreachable!(),
        }
    }

    fn calc_sum(x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
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

    fn calc_diff(x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
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
