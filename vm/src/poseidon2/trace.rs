use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::{columns::*, Poseidon2Chip};

impl<const WIDTH: usize, const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    Poseidon2Chip<WIDTH, NUM_WORDS, WORD_SIZE, F>
{
    /// Generates final Poseidon2VmAir trace from cached rows.
    pub fn generate_trace(&self) -> RowMajorMatrix<F> {
        println!(
            "in generate_trace, WIDTH, NUM_WORDS, WORD_SIZE: {}, {}, {}",
            WIDTH, NUM_WORDS, WORD_SIZE
        );

        let row_len = self.rows.len();
        let correct_len = row_len.next_power_of_two();
        let diff = correct_len - row_len;

        let mut flat_rows = Vec::with_capacity(correct_len * self.air.width());
        for row in self.rows.iter() {
            println!("non-blank row width: {}", row.clone().flatten().len());
            println!(
                "width(): {}",
                Poseidon2VmCols::<WIDTH, WORD_SIZE, F>::width(&self.air)
            );
            println!("num of mem_access: {}", row.aux.mem_oc_aux_cols.len());
            println!("3 + 2 * WIDTH: {}", 3 + 2 * WIDTH);
            flat_rows.extend(row.flatten());
        }

        for _ in 0..diff {
            flat_rows.extend(self.blank_row().flatten());
        }

        println!("correct_len: {}", correct_len);

        RowMajorMatrix::new(flat_rows, self.air.width())
    }

    pub fn blank_row(&self) -> Poseidon2VmCols<WIDTH, WORD_SIZE, F> {
        let mut blank = Poseidon2VmCols::<WIDTH, WORD_SIZE, F>::blank_row(&self.air.inner);

        // TODO[osama]: think whether clk should start at zero or one
        let mut clk = F::one();
        for _ in 0..3 {
            blank.aux.mem_oc_aux_cols.push(
                self.air.mem_oc.disabled_memory_checker_aux_cols_from_op(
                    blank.io.d,
                    clk,
                    self.range_checker.clone(),
                ),
            );
            clk += F::one();
        }
        for _ in 0..2 * WIDTH {
            blank.aux.mem_oc_aux_cols.push(
                self.air.mem_oc.disabled_memory_checker_aux_cols_from_op(
                    blank.io.e,
                    clk,
                    self.range_checker.clone(),
                ),
            );
            clk += F::one();
        }

        blank
    }
}
