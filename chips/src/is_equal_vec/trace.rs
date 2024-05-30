use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::IsEqualVecChip;

impl IsEqualVecChip {
    pub fn generate_trace_rows<F: PrimeField32>(&self) -> RowMajorMatrix<F> {
        let width: usize = self.get_width();
        let vec_len: usize = self.vec_len();
        let height: usize = self.x.len();
        assert!(height.is_power_of_two());
        let mut rows = Vec::with_capacity(height);

        // TODO make sexy
        for i in 0..height {
            let mut row = self.x[i]
                .iter()
                .chain(self.y[i].iter())
                .map(|&val| F::from_canonical_u32(val))
                .chain(std::iter::repeat(F::one()).take(2 * vec_len))
                .collect::<Vec<F>>();

            let mut broken = false;
            let mut post_broken = false;

            for j in 0..vec_len {
                if row[j] != row[j + vec_len] || broken {
                    row[j + 2 * vec_len] = F::zero();
                    broken = true;
                }

                if !post_broken {
                    row[j + 3 * vec_len] =
                        (row[j] - row[j + vec_len] + row[j + 2 * vec_len]).inverse();
                } else {
                    row[j + 3 * vec_len] = F::zero();
                }

                if broken {
                    post_broken = true;
                }
            }

            rows.push(row.clone());
        }

        RowMajorMatrix::new(rows.concat(), width)
    }
}
