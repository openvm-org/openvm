use p3_field::{Field, PrimeField64};
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
    sum: &[u32],
    carry: &[u32],
) -> Vec<T> {
    let base_op_u8 = OpCode::ADD256 as u8;
    LongArithmeticCols::<ARG_SIZE, LIMB_SIZE, T> {
        io: LongArithmeticIoCols {
            rcv_count: T::one(),
            opcode: T::from_canonical_u8(opcode as u8),
            x_limbs: x.iter().map(|x| T::from_canonical_u32(*x)).collect(),
            y_limbs: y.iter().map(|x| T::from_canonical_u32(*x)).collect(),
            z_limbs: sum.iter().map(|x| T::from_canonical_u32(*x)).collect(),
        },
        aux: LongArithmeticAuxCols {
            opcode_sub_flag: T::from_canonical_u8(opcode as u8 - base_op_u8),
            carry: carry.iter().map(|x| T::from_canonical_u32(*x)).collect(),
        },
    }
    .flatten()
}

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongArithmeticChip<ARG_SIZE, LIMB_SIZE> {
    // return the sum and the carry
    fn calculate(opcode: OpCode, x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
        match opcode {
            OpCode::ADD256 => Self::calc_sum(x, y),
            OpCode::SUB256 => Self::calc_diff(x, y),
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

    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .operations
            .iter()
            .map(|operation: &LongArithmeticOperation| {
                let (opcode, x, y) = (operation.opcode, &operation.operand1, &operation.operand2);
                let (sum, carry) = Self::calculate(opcode, x, y);
                assert!(ARG_SIZE % LIMB_SIZE == 0);
                for z in &sum {
                    self.range_checker_chip.add_count(*z);
                }
                create_row_from_values::<ARG_SIZE, LIMB_SIZE, F>(opcode, x, y, &sum, &carry)
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
        );
        // set rcv_count to 0
        let blank_row = [vec![F::zero()], blank_row[1..].to_vec()].concat();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
