use std::{collections::HashSet, sync::Arc};

use afs_primitives::{
    is_less_than_tuple::columns::IsLessThanTupleCols, range_gate::RangeCheckerGateChip,
    sub_chip::LocalTraceInstructions,
};
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::info_span;

use crate::{common::page::Page, multitier_page_rw_checker::leaf_page_air::PageRwAir};

use super::LeafPageAir;

impl<const COMMITMENT_LEN: usize> LeafPageAir<COMMITMENT_LEN> {
    // The trace is the whole page (including the is_alloc column)
    pub fn generate_cached_trace_from_page<F: PrimeField64>(
        &self,
        page: &Page,
    ) -> RowMajorMatrix<F> {
        page.gen_trace()
    }

    pub fn generate_cached_trace_from_2d_vec<F: PrimeField64>(
        &self,
        page: &[Vec<u32>],
    ) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            page.iter()
                .flat_map(|row| {
                    row.iter()
                        .map(|n: &u32| F::from_wrapped_u32(*n))
                        .collect::<Vec<F>>()
                })
                .collect(),
            self.cached_width(),
        )
    }

    pub fn generate_main_trace<SC: StarkGenericConfig>(
        &self,
        page: Page,
        range: (Vec<u32>, Vec<u32>),
        range_checker: Arc<RangeCheckerGateChip>,
        internal_indices: &HashSet<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField64 + PrimeField,
    {
        let mut final_page_aux_rows = match &self.page_chip {
            PageRwAir::Final(fin) => {
                tracing::info_span!("Final Page Trace Generation").in_scope(|| {
                    fin.gen_aux_trace::<SC>(&page, range_checker.clone(), internal_indices)
                })
            }
            _ => RowMajorMatrix::new(vec![], 1),
        };
        RowMajorMatrix::new(
            info_span!("Less Than Trace Generation").in_scope(|| {
                page.iter()
                    .enumerate()
                    .flat_map(|(i, row)| {
                        let mut trace_row = vec![];
                        if !self.is_init {
                            trace_row.extend(range.0.clone());
                            trace_row.extend(range.1.clone());
                            trace_row.extend(vec![0; 2]);
                            let mut trace_row: Vec<Val<SC>> = trace_row
                                .into_iter()
                                .map(Val::<SC>::from_canonical_u32)
                                .collect();
                            {
                                let tuple: IsLessThanTupleCols<Val<SC>> =
                                    LocalTraceInstructions::generate_trace_row(
                                        &self.is_less_than_tuple_air.as_ref().unwrap().idx_start,
                                        (row.idx.to_vec(), range.0.clone(), range_checker.clone()),
                                    );
                                let aux = tuple.aux;
                                let io = tuple.io;
                                trace_row[2 * range.0.len()] = io.tuple_less_than;
                                trace_row.extend(aux.flatten());
                            }
                            {
                                let tuple: IsLessThanTupleCols<Val<SC>> =
                                    LocalTraceInstructions::generate_trace_row(
                                        &self.is_less_than_tuple_air.as_ref().unwrap().end_idx,
                                        (range.1.clone(), row.idx.to_vec(), range_checker.clone()),
                                    );
                                let aux = tuple.aux;
                                let io = tuple.io;
                                trace_row[2 * range.0.len() + 1] = io.tuple_less_than;
                                trace_row.extend(aux.flatten());
                            }
                            {
                                trace_row.extend(final_page_aux_rows.row_mut(i).to_vec());
                            }
                            trace_row
                        } else {
                            trace_row
                                .into_iter()
                                .map(Val::<SC>::from_wrapped_u32)
                                .collect::<Vec<Val<SC>>>()
                        }
                    })
                    .collect()
            }),
            self.main_width(),
        )
    }
}
