use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::LongMultiplicationCols, num_limbs, LongMultiplicationChip};
use crate::cpu::OpCode;

struct CalculationResult {
    op: OpCode,
    z_limbs: Vec<u32>,
    carry: Vec<u32>,
}

impl LongMultiplicationChip {
    fn calculate(&self, op: OpCode, x: &[u32], y: &[u32]) -> CalculationResult {
        let num_limbs = num_limbs(self.arg_size, self.limb_size);
        assert!(x.len() == num_limbs && y.len() == num_limbs);

        let mut z_limbs = vec![0; num_limbs];
        let mut carry = vec![0; num_limbs];

        for i in 0..num_limbs {
            let sum = (0..=i).map(|j| x[j] * y[i - j]).sum::<u32>()
                + if i > 0 { carry[i - 1] } else { 0 };

            z_limbs[i] = sum & ((1 << self.limb_size) - 1);
            carry[i] = sum >> self.limb_size;
        }

        CalculationResult { op, z_limbs, carry }
    }

    pub fn generate_trace<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .operations
            .iter()
            .map(|operation| {
                let (opcode, x, y) = (
                    operation.opcode,
                    &operation.multiplicand,
                    &operation.multiplier,
                );
                let CalculationResult { op, z_limbs, carry } = self.calculate(opcode, x, y);
                let num_limbs = num_limbs(self.arg_size, self.limb_size);
                for z in z_limbs.iter() {
                    // TODO: replace with a more optimal range check once we have one
                    self.range_checker_chip.add_count(*z);
                    self.range_checker_chip
                        .add_count(*z + ((num_limbs - 1) << self.limb_size) as u32);
                }
                for c in carry.iter() {
                    self.range_checker_chip.add_count(*c);
                }
                LongMultiplicationCols::<F> {
                    rcv_count: F::one(),
                    opcode: F::from_canonical_u8(op as u8),
                    x_limbs: x.iter().map(|x| F::from_canonical_u32(*x)).collect(),
                    y_limbs: y.iter().map(|y| F::from_canonical_u32(*y)).collect(),
                    z_limbs: z_limbs.iter().map(|z| F::from_canonical_u32(*z)).collect(),
                    carry: carry.iter().map(|c| F::from_canonical_u32(*c)).collect(),
                }
                .flatten()
            })
            .collect::<Vec<_>>();

        let height = rows.len();
        let padded_height = height.next_power_of_two();

        let num_limbs = num_limbs(self.arg_size, self.limb_size);

        let blank_row = LongMultiplicationCols::<F> {
            rcv_count: F::zero(),
            opcode: F::from_canonical_u8(self.mul_op as u8),
            x_limbs: vec![F::zero(); num_limbs],
            y_limbs: vec![F::zero(); num_limbs],
            z_limbs: vec![F::zero(); num_limbs],
            carry: vec![F::zero(); num_limbs],
        }
        .flatten();
        let width = blank_row.len();

        let mut padded_rows = rows;
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
