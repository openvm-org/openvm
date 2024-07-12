use p3_matrix::dense::RowMajorMatrix;

use p3_field::PrimeField64;

use super::RootSignalAir;

impl<const COMMITMENT_LEN: usize> RootSignalAir<COMMITMENT_LEN> {
    pub fn generate_trace<F: PrimeField64>(
        &self,
        commit: Vec<u32>,
        id: u32,
        mult: u32,
        range: (Vec<u32>, Vec<u32>),
    ) -> RowMajorMatrix<F> {
        assert!(commit.len() == COMMITMENT_LEN);
        RowMajorMatrix::new(
            {
                let mut trace_row = vec![];
                trace_row.extend(commit.clone());
                trace_row.push(id);
                trace_row.push(mult);
                if !self.is_init {
                    trace_row.extend(range.0.clone());
                    trace_row.extend(range.1.clone());
                    trace_row
                        .into_iter()
                        .map(|i| F::from_wrapped_u32(i))
                        .collect::<Vec<F>>()
                } else {
                    trace_row
                        .into_iter()
                        .map(|i| F::from_wrapped_u32(i))
                        .collect::<Vec<F>>()
                }
            },
            self.air_width(),
        )
    }
}
