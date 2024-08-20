use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{LongArithmeticAuxCols, LongArithmeticCols, LongArithmeticIoCols},
    num_limbs, LongArithmeticChip, LongArithmeticOperation,
};
use crate::cpu::OpCode;

pub fn create_row_from_values<const ARG_SIZE: usize, const LIMB_SIZE: usize, T: Field>(
    opcode: OpCode,
    x: &[u32],
    y: &[u32],
    result_limbs: &[u32],
    buffer_limbs: &[u32],
    cmp_result: bool,
) -> Vec<T> {
    LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, T> {
        io: LongArithmeticIoCols {
            rcv_count: T::one(),
            opcode: T::from_canonical_u8(opcode as u8),
            x_limbs: x.iter().map(|x| T::from_canonical_u32(*x)).collect(),
            y_limbs: y.iter().map(|x| T::from_canonical_u32(*x)).collect(),
            z_limbs: result_limbs
                .iter()
                .map(|x| T::from_canonical_u32(*x))
                .collect(),
            cmp_result: T::from_canonical_u8(cmp_result as u8),
        },
        aux: LongArithmeticAuxCols {
            opcode_add_flag: T::from_canonical_u8((opcode == OpCode::ADD256) as u8),
            opcode_sub_flag: T::from_canonical_u8((opcode == OpCode::SUB256) as u8),
            opcode_lt_flag: T::from_canonical_u8((opcode == OpCode::LT256) as u8),
            opcode_eq_flag: T::from_canonical_u8((opcode == OpCode::EQ256) as u8),
            buffer: buffer_limbs
                .iter()
                .map(|x| T::from_canonical_u32(*x))
                .collect(),
        },
    }
    .flatten()
}

struct CalculationResult {
    result_limbs: Vec<u32>,
    buffer_limbs: Vec<u32>,
    cmp_result: bool,
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticChip<ARG_SIZE, LIMB_SIZE> {
    fn calculate<F: PrimeField32>(opcode: OpCode, x: &[u32], y: &[u32]) -> CalculationResult {
        match opcode {
            OpCode::ADD256 => {
                let (sum, carry) = Self::calc_sum(x, y);
                CalculationResult {
                    result_limbs: sum,
                    buffer_limbs: carry,
                    cmp_result: false,
                }
            }
            OpCode::SUB256 => {
                let (diff, carry) = Self::calc_diff(x, y);
                CalculationResult {
                    result_limbs: diff,
                    buffer_limbs: carry,
                    cmp_result: false,
                }
            }
            OpCode::LT256 => {
                let (diff, carry) = Self::calc_diff(x, y);
                let cmp_result = *carry.last().unwrap() == 1;
                CalculationResult {
                    result_limbs: diff,
                    buffer_limbs: carry,
                    cmp_result,
                }
            }
            OpCode::EQ256 => {
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
                CalculationResult {
                    result_limbs: vec![0u32; num_limbs],
                    buffer_limbs: inverse,
                    cmp_result: x.iter().zip(y).all(|(x, y)| x == y),
                }
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

    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .operations
            .iter()
            .map(|operation: &LongArithmeticOperation| {
                let (opcode, x, y) = (operation.opcode, &operation.operand1, &operation.operand2);
                let CalculationResult {
                    result_limbs,
                    buffer_limbs,
                    cmp_result,
                } = Self::calculate::<F>(opcode, x, y);
                assert!(ARG_SIZE % LIMB_SIZE == 0);
                if opcode == OpCode::ADD256 || opcode == OpCode::SUB256 || opcode == OpCode::LT256 {
                    for z in &result_limbs {
                        self.range_checker_chip.add_count(*z);
                    }
                }
                create_row_from_values::<ARG_SIZE, LIMB_SIZE, F>(
                    opcode,
                    x,
                    y,
                    &result_limbs,
                    &buffer_limbs,
                    cmp_result,
                )
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let num_limbs = num_limbs::<ARG_SIZE, LIMB_SIZE>();

        let blank_row = create_row_from_values::<ARG_SIZE, LIMB_SIZE, F>(
            OpCode::ADD256,
            &vec![0u32; num_limbs],
            &vec![0u32; num_limbs],
            &vec![0u32; num_limbs],
            &vec![0u32; num_limbs],
            false,
        );
        // set rcv_count to 0
        let blank_row = [vec![F::zero()], blank_row[1..].to_vec()].concat();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
