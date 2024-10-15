use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use super::VolatileBoundaryChip;
use crate::system::memory::{volatile::columns::VolatileBoundaryCols, TimestampedEquipartition};

impl<F: PrimeField32> VolatileBoundaryChip<F> {
    pub fn generate_trace(
        &self,
        final_memory: &TimestampedEquipartition<F, 1>,
    ) -> RowMajorMatrix<F> {
        let trace_height = final_memory.len().next_power_of_two();
        self.generate_trace_with_height(final_memory, trace_height)
    }

    pub fn generate_trace_with_height(
        &self,
        final_memory: &TimestampedEquipartition<F, 1>,
        trace_height: usize,
    ) -> RowMajorMatrix<F> {
        assert!(trace_height.is_power_of_two());

        let gen_row = |prev_idx: Vec<F>,
                       cur_idx: Vec<F>,
                       final_data: F,
                       final_timestamp: F,
                       initial_data: F,
                       is_extra: F| {
            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.air.addr_lt_air,
                (
                    prev_idx.iter().map(|x| x.as_canonical_u32()).collect(),
                    cur_idx.iter().map(|x| x.as_canonical_u32()).collect(),
                    self.range_checker.clone(),
                ),
            );

            VolatileBoundaryCols::<F>::new(
                cur_idx[0],
                cur_idx[1],
                initial_data,
                final_data,
                final_timestamp,
                is_extra,
                lt_cols.io.tuple_less_than,
                lt_cols.aux,
            )
        };

        let mut rows_concat = Vec::with_capacity(trace_height * self.air.air_width());
        let mut prev_idx = vec![F::zero(), F::zero()];
        for ((address_space, label), timedstamped_values) in final_memory.iter() {
            let cur_idx = vec![*address_space, F::from_canonical_usize(*label)];

            let [data] = timedstamped_values.values;

            rows_concat.extend(
                gen_row(
                    prev_idx,
                    cur_idx.clone(),
                    data,
                    F::from_canonical_u32(timedstamped_values.timestamp),
                    F::zero(),
                    F::zero(),
                )
                .flatten(),
            );

            prev_idx = cur_idx;
        }

        assert!(rows_concat.len() <= trace_height * self.air.air_width());

        let dummy_idx = vec![F::zero(), F::zero()];
        let dummy_data = F::zero();
        let dummy_clk = F::zero();

        while rows_concat.len() < trace_height * self.air.air_width() {
            rows_concat.extend(
                gen_row(
                    prev_idx.clone(),
                    dummy_idx.clone(),
                    dummy_data,
                    dummy_clk,
                    dummy_data,
                    F::one(),
                )
                .flatten(),
            );

            prev_idx.clone_from(&dummy_idx);
        }

        RowMajorMatrix::new(rows_concat, self.air.air_width())
    }
}
