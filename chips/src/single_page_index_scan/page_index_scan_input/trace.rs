use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{common::page::Page, sub_chip::LocalTraceInstructions};

use super::{
    EqCompAir, NonStrictCompAir, PageIndexScanInputAirVariants, PageIndexScanInputChip,
    StrictCompAir,
};

impl PageIndexScanInputChip {
    /// Generate the trace for the page table
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        page.gen_trace()
    }

    /// Helper function to handle trace generation with an IsLessThanTupleAir
    fn handle_is_less_than_tuple<SC: StarkGenericConfig>(
        &self,
        is_less_than_tuple_trace: Vec<Val<SC>>,
        is_alloc: Val<SC>,
        row: &mut Vec<Val<SC>>,
    ) where
        Val<SC>: AbstractField + PrimeField64,
    {
        // satisfies_pred, send_row, is_less_than_tuple_aux_cols
        row.push(is_less_than_tuple_trace[2 * self.air.idx_len]);
        let send_row = is_less_than_tuple_trace[2 * self.air.idx_len] * is_alloc;
        row.push(send_row);

        row.extend_from_slice(&is_less_than_tuple_trace[2 * self.air.idx_len + 1..]);
    }

    /// Helper function to handle trace generation with an IsEqualVecAir
    fn handle_is_equal_vec<SC: StarkGenericConfig>(
        &self,
        is_equal_vec_trace: Vec<Val<SC>>,
        is_alloc: Val<SC>,
        row: &mut Vec<Val<SC>>,
    ) where
        Val<SC>: AbstractField + PrimeField64,
    {
        // satisfies_pred, send_row, is_equal_vec_aux_cols
        row.push(is_equal_vec_trace[2 * self.air.idx_len]);
        let send_row = is_equal_vec_trace[2 * self.air.idx_len] * is_alloc;
        row.push(send_row);

        row.extend_from_slice(&is_equal_vec_trace[2 * self.air.idx_len + 1..]);
    }

    /// Helper function to handle trace generation with an IsLessThanTupleAir and an IsEqualVecAir
    fn handle_both_airs<SC: StarkGenericConfig>(
        &self,
        is_less_than_tuple_trace: Vec<Val<SC>>,
        is_equal_vec_trace: Vec<Val<SC>>,
        is_alloc: Val<SC>,
        row: &mut Vec<Val<SC>>,
    ) where
        Val<SC>: AbstractField + PrimeField64,
    {
        // satisfies_pred, send_row, satisfies_strict_comp, satisfies_eq_comp, is_less_than_tuple_aux_cols, is_equal_vec_aux_cols
        let satisfies_pred = is_less_than_tuple_trace[2 * self.air.idx_len]
            + is_equal_vec_trace[2 * self.air.idx_len];
        row.push(satisfies_pred);
        row.push(satisfies_pred * is_alloc);

        row.push(is_less_than_tuple_trace[2 * self.air.idx_len]);
        row.push(is_equal_vec_trace[2 * self.air.idx_len]);

        row.extend_from_slice(&is_less_than_tuple_trace[2 * self.air.idx_len + 1..]);
        row.extend_from_slice(&is_equal_vec_trace[2 * self.air.idx_len + 1..]);
    }

    /// Generate the trace for the auxiliary columns
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        x: Vec<u32>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        let mut rows: Vec<Val<SC>> = vec![];

        for page_row in page.iter() {
            let mut row: Vec<Val<SC>> = vec![];

            let is_alloc = Val::<SC>::from_canonical_u32(page_row.is_alloc);
            let idx = page_row.idx.clone();

            // first, get the values for x
            let x_trace: Vec<Val<SC>> = x
                .iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect();
            row.extend(x_trace);

            let is_less_than_tuple_trace: Option<Vec<Val<SC>>> = match &self.air.variant_air {
                PageIndexScanInputAirVariants::Lt(StrictCompAir {
                    is_less_than_tuple_air,
                    ..
                })
                | PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                    is_less_than_tuple_air,
                    ..
                }) => Some(
                    LocalTraceInstructions::generate_trace_row(
                        is_less_than_tuple_air,
                        (idx.clone(), x.clone(), self.range_checker.clone()),
                    )
                    .flatten(),
                ),
                PageIndexScanInputAirVariants::Gt(StrictCompAir {
                    is_less_than_tuple_air,
                    ..
                })
                | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                    is_less_than_tuple_air,
                    ..
                }) => Some(
                    is_less_than_tuple_air
                        .generate_trace_row((x.clone(), idx.clone(), self.range_checker.clone()))
                        .flatten(),
                ),
                _ => None,
            };

            let is_equal_vec_trace: Option<Vec<Val<SC>>> = match &self.air.variant_air {
                PageIndexScanInputAirVariants::Lte(NonStrictCompAir {
                    is_equal_vec_air, ..
                })
                | PageIndexScanInputAirVariants::Eq(EqCompAir {
                    is_equal_vec_air, ..
                })
                | PageIndexScanInputAirVariants::Gte(NonStrictCompAir {
                    is_equal_vec_air, ..
                }) => Some(
                    is_equal_vec_air
                        .generate_trace_row((
                            idx.clone()
                                .into_iter()
                                .map(Val::<SC>::from_canonical_u32)
                                .collect(),
                            x.clone()
                                .into_iter()
                                .map(Val::<SC>::from_canonical_u32)
                                .collect(),
                        ))
                        .flatten(),
                ),
                _ => None,
            };

            match &self.air.variant_air {
                PageIndexScanInputAirVariants::Lt(..) | PageIndexScanInputAirVariants::Gt(..) => {
                    self.handle_is_less_than_tuple::<SC>(
                        is_less_than_tuple_trace.unwrap(),
                        is_alloc,
                        &mut row,
                    );
                }
                PageIndexScanInputAirVariants::Lte(..) | PageIndexScanInputAirVariants::Gte(..) => {
                    self.handle_both_airs::<SC>(
                        is_less_than_tuple_trace.unwrap(),
                        is_equal_vec_trace.unwrap(),
                        is_alloc,
                        &mut row,
                    );
                }
                PageIndexScanInputAirVariants::Eq(..) => {
                    self.handle_is_equal_vec::<SC>(is_equal_vec_trace.unwrap(), is_alloc, &mut row);
                }
            }

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, self.aux_width())
    }
}
