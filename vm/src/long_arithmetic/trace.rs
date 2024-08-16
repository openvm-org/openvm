use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::LongAdditionCols, LongAdditionChip};

impl<const ARG_SIZE: usize, const LIMB_SIZE: usize> LongAdditionChip<ARG_SIZE, LIMB_SIZE> {
    // TODO: move it to the proper place once we add opcodes and stuff
    // return the sum and the carry
    fn calc_sum(x: &[u32], y: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let num_limbs = (ARG_SIZE + LIMB_SIZE - 1) / LIMB_SIZE; // TODO: bad duplication, maybe move somewhere else
        let mut result = vec![0u32; num_limbs];
        let mut carry = vec![0u32; num_limbs];
        for i in 0..num_limbs {
            result[i] = x[i] + y[i] + if i > 0 { carry[i - 1] } else { 0 };
            carry[i] = result[i] >> LIMB_SIZE;
            result[i] &= (1 << LIMB_SIZE) - 1;
        }
        (result, carry)
    }

    pub fn generate_trace<F: PrimeField64>(&self) -> RowMajorMatrix<F> {
        let rows = self
            .operations
            .iter()
            .map(|(x, y)| {
                let (sum, carry) = Self::calc_sum(x, y);
                assert!(ARG_SIZE % LIMB_SIZE == 0);
                for z in &sum {
                    self.range_checker_chip.add_count(*z);
                }
                [x, y, &sum, &carry]
                    .into_iter()
                    .flatten()
                    .map(|x| F::from_canonical_u32(*x))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let width = LongAdditionCols::<ARG_SIZE, LIMB_SIZE, F>::get_width();
        let height = rows.len();
        let padded_height = height.next_power_of_two();

        for _ in
            0..(padded_height - height) * LongAdditionCols::<ARG_SIZE, LIMB_SIZE, F>::num_limbs()
        {
            self.range_checker_chip.add_count(0);
        }

        let mut padded_rows = rows;
        let blank_row = vec![F::zero(); width];
        padded_rows.extend(std::iter::repeat(blank_row).take(padded_height - height));

        RowMajorMatrix::new(padded_rows.concat(), width)
    }
}
