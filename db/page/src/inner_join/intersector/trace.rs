use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use afs_primitives::{range_gate::RangeCheckerGateChip, sub_chip::LocalTraceInstructions};
use afs_test_utils::utils::to_field_vec;
use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{IntersectorAuxCols, IntersectorCols, IntersectorIoCols},
    IntersectorAir,
};
use crate::common::page::Page;

impl IntersectorAir {
    pub fn generate_trace<F: PrimeField>(
        &mut self,
        t1: &Page,
        t2: &Page,
        fkey_start: usize,
        fkey_end: usize,
        range_checker: Arc<RangeCheckerGateChip>,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        let mut t1_idx_mult = HashMap::new();
        let mut t2_idx_mult = HashMap::new();
        let mut all_indices = HashSet::new();

        for row in t1.iter() {
            if row.is_alloc == 0 {
                continue;
            }
            *t1_idx_mult.entry(row.idx.clone()).or_insert(0) += 1;
            all_indices.insert(row.idx.clone());
        }

        for row in t2.iter() {
            if row.is_alloc == 0 {
                continue;
            }

            let fkey = row.data[fkey_start..fkey_end].to_vec();
            *t2_idx_mult.entry(fkey.clone()).or_insert(0) += 1;
            all_indices.insert(fkey);
        }

        let mut all_indices: Vec<Vec<u32>> = all_indices.into_iter().collect();
        all_indices.sort();

        let mut rows: Vec<Vec<F>> = vec![];
        let mut prv_idx = vec![0; self.idx_len];
        for idx in all_indices {
            let t1_mult = *t1_idx_mult.get(&idx).unwrap_or(&0);
            let t2_mult = *t2_idx_mult.get(&idx).unwrap_or(&0);
            let out_mult = t1_mult * t2_mult;

            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.lt_chip,
                (prv_idx.clone(), idx.clone(), range_checker.clone()),
            );

            prv_idx.clone_from(&idx);

            let inter_cols = IntersectorCols {
                io: IntersectorIoCols {
                    idx: to_field_vec::<F>(idx),
                    t1_mult: F::from_canonical_u32(t1_mult),
                    t2_mult: F::from_canonical_u32(t2_mult),
                    out_mult: F::from_canonical_u32(out_mult),
                    is_extra: F::zero(),
                },
                aux: IntersectorAuxCols {
                    lt_aux: lt_cols.aux,
                    lt_out: lt_cols.io.tuple_less_than,
                },
            };

            rows.push(inter_cols.flatten());
        }

        // Padding the trace to be of degree trace_degree
        rows.resize_with(trace_degree, || {
            let lt_cols = LocalTraceInstructions::generate_trace_row(
                &self.lt_chip,
                (
                    prv_idx.clone(),
                    vec![0; self.idx_len],
                    range_checker.clone(),
                ),
            );

            prv_idx = vec![0; self.idx_len];

            let inter_cols = IntersectorCols {
                io: IntersectorIoCols {
                    idx: vec![F::zero(); self.idx_len],
                    t1_mult: F::zero(),
                    t2_mult: F::zero(),
                    out_mult: F::zero(),
                    is_extra: F::one(),
                },
                aux: IntersectorAuxCols {
                    lt_aux: lt_cols.aux,
                    lt_out: lt_cols.io.tuple_less_than,
                },
            };

            inter_cols.flatten()
        });

        RowMajorMatrix::new(rows.concat(), self.air_width())
    }
}
