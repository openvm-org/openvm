use std::{
    collections::{HashMap, HashSet},
    iter,
};

use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use crate::common::page::Page;

use super::IntersectorAir;

impl IntersectorAir {
    pub fn generate_trace<F: PrimeField>(
        &mut self,
        t1: &Page,
        t2: &Page,
        fkey_start: usize,
        fkey_end: usize,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        let mut t1_idx_mult = HashMap::new();
        let mut t2_idx_mult = HashMap::new();
        let mut all_indices = HashSet::new();

        for row in t1.rows.iter() {
            if row.is_alloc == 0 {
                continue;
            }
            *t1_idx_mult.entry(row.idx.clone()).or_insert(0) += 1;
            all_indices.insert(row.idx.clone());
        }

        for row in t2.rows.iter() {
            if row.is_alloc == 0 {
                continue;
            }

            let fkey = row.data[fkey_start..fkey_end].to_vec();
            *t2_idx_mult.entry(fkey.clone()).or_insert(0) += 1;
            all_indices.insert(fkey);
        }

        let mut rows: Vec<Vec<F>> = vec![];
        for idx in all_indices {
            let t1_mult = *t1_idx_mult.get(&idx).unwrap();
            let t2_mult = *t2_idx_mult.get(&idx).unwrap();
            let out_mult = t1_mult * t2_mult;

            rows.push(
                idx.iter()
                    .copied()
                    .chain(iter::once(t1_mult))
                    .chain(iter::once(t2_mult))
                    .chain(iter::once(out_mult))
                    .chain(iter::once(0)) // non-extra row
                    .map(F::from_canonical_u32)
                    .collect(),
            );
        }

        let width = t1.idx_len() + 4;
        rows.resize_with(trace_degree, || vec![F::one(); width]);

        RowMajorMatrix::new(rows.concat(), width)
    }
}
