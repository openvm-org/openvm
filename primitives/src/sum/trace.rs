use p3_air::BaseAir;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use super::SumChip;
use crate::{is_less_than::columns::IsLessThanCols, sub_chip::LocalTraceInstructions};

impl SumChip {
    pub fn generate_trace<F: PrimeField64>(&self, inputs: &[(u32, u32)]) -> RowMajorMatrix<F> {
        let n = inputs.len();
        assert!(n.is_power_of_two());

        let mut sorted_inputs = inputs.to_vec();
        sorted_inputs.sort_by(|a, b| a.0.cmp(&b.0));

        let mut rows: Vec<Vec<F>> = Vec::with_capacity(n);
        let mut partial_sum = 0;

        for (i, &(key, value)) in sorted_inputs.iter().enumerate() {
            partial_sum += value;

            let is_final = i == n - 1 || key != sorted_inputs[i + 1].0;
            let mut row: Vec<F> = vec![key, value, partial_sum, is_final as u32]
                .into_iter()
                .map(F::from_canonical_u32)
                .collect();

            // for the final row, there is no is_lt check, so we can use whatever for next_key;
            // wrapping around seems easiest
            let next_key = sorted_inputs[(i + 1) % n].0;
            let is_less_than_row: IsLessThanCols<F> = LocalTraceInstructions::generate_trace_row(
                &self.air.is_lt_air,
                (key, next_key, self.range_checker.clone()),
            );

            row.extend(is_less_than_row.aux.flatten());

            rows.push(row);

            if is_final {
                partial_sum = 0;
            }
        }

        let width = BaseAir::<F>::width(&self.air);
        RowMajorMatrix::new(rows.concat(), width)
    }
}
