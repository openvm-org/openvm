use afs_primitives::sub_chip::LocalTraceInstructions;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::FilterInputTableChip;
use crate::{
    common::page::Page,
    filter::input_table::{air::FilterAirVariants, columns::FilterInputTableAuxCols},
};

impl FilterInputTableChip {
    pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        page.gen_trace()
    }

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
            let select_cols =
                page_row.to_vec()[self.air.start_col + 1..self.air.end_col + 1].to_vec();
            let select_cols_f: Vec<Val<SC>> = select_cols
                .iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect();

            // first, get the values for x
            let x_trace: Vec<Val<SC>> = x
                .iter()
                .map(|x| Val::<SC>::from_canonical_u32(*x))
                .collect();
            row.extend(x_trace.clone());

            let is_less_than_tuple_trace: Option<Vec<Val<SC>>> = match &self.air.variant_air {
                FilterAirVariants::Lt(strict_comp_air) => Some(
                    LocalTraceInstructions::generate_trace_row(
                        &strict_comp_air.is_less_than_tuple_air,
                        (select_cols.clone(), x.clone(), self.range_checker.clone()),
                    )
                    .flatten(),
                ),
                FilterAirVariants::Gt(strict_comp_air) => Some(
                    LocalTraceInstructions::generate_trace_row(
                        &strict_comp_air.is_less_than_tuple_air,
                        (x.clone(), select_cols.clone(), self.range_checker.clone()),
                    )
                    .flatten(),
                ),
                _ => None,
            };

            let is_not_less_than_tuple_trace: Option<Vec<Val<SC>>> = match &self.air.variant_air {
                FilterAirVariants::Lte(strict_inv_comp_air) => {
                    let mut row = LocalTraceInstructions::generate_trace_row(
                        &strict_inv_comp_air.is_less_than_tuple_air,
                        (x.clone(), select_cols.clone(), self.range_checker.clone()),
                    )
                    .flatten();
                    row.push(Val::<SC>::from_canonical_u32(
                        strict_inv_comp_air.inv as u32,
                    ));
                    Some(row)
                }
                FilterAirVariants::Gte(strict_inv_comp_air) => {
                    let mut row = LocalTraceInstructions::generate_trace_row(
                        &strict_inv_comp_air.is_less_than_tuple_air,
                        (select_cols.clone(), x.clone(), self.range_checker.clone()),
                    )
                    .flatten();
                    row.push(Val::<SC>::from_canonical_u32(
                        strict_inv_comp_air.inv as u32,
                    ));
                    Some(row)
                }
                _ => None,
            };

            let is_equal_vec_trace: Option<Vec<Val<SC>>> = match &self.air.variant_air {
                FilterAirVariants::Eq(eq_comp_air) => Some(
                    LocalTraceInstructions::generate_trace_row(
                        &eq_comp_air.is_equal_vec_air,
                        (select_cols_f.clone(), x_trace.clone()),
                    )
                    .flatten(),
                ),
                _ => None,
            };

            match &self.air.variant_air {
                FilterAirVariants::Lt(..) | FilterAirVariants::Gt(..) => {
                    self.handle_is_less_than_tuple::<SC>(
                        is_less_than_tuple_trace.unwrap(),
                        is_alloc,
                        &mut row,
                    );
                }
                FilterAirVariants::Lte(..) | FilterAirVariants::Gte(..) => {
                    self.handle_is_not_less_than_tuple::<SC>(
                        is_not_less_than_tuple_trace.unwrap(),
                        is_alloc,
                        &mut row,
                    );
                }
                FilterAirVariants::Eq(..) => {
                    self.handle_is_equal_vec::<SC>(is_equal_vec_trace.unwrap(), is_alloc, &mut row);
                }
            }

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, self.air.aux_width())
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
        row.push(is_less_than_tuple_trace[2 * self.air.num_filter_cols()]);
        let send_row = is_less_than_tuple_trace[2 * self.air.num_filter_cols()] * is_alloc;
        row.push(send_row);

        row.extend_from_slice(&is_less_than_tuple_trace[2 * self.air.num_filter_cols() + 1..]);
    }

    /// Helper function to handle trace generation with a NOT IsLessThanTupleAir
    fn handle_is_not_less_than_tuple<SC: StarkGenericConfig>(
        &self,
        is_not_less_than_tuple_trace: Vec<Val<SC>>,
        is_alloc: Val<SC>,
        row: &mut Vec<Val<SC>>,
    ) where
        Val<SC>: AbstractField + PrimeField64,
    {
        // satisfies_pred, send_row, is_less_than_tuple_aux_cols
        row.push(is_not_less_than_tuple_trace[2 * self.air.num_filter_cols()]);
        let send_row = is_not_less_than_tuple_trace[2 * self.air.num_filter_cols()] * is_alloc;
        row.push(send_row);

        row.extend_from_slice(&is_not_less_than_tuple_trace[2 * self.air.num_filter_cols() + 1..]);
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
        row.push(is_equal_vec_trace[2 * self.air.num_filter_cols()]);
        let send_row = is_equal_vec_trace[2 * self.air.num_filter_cols()] * is_alloc;
        row.push(send_row);

        row.extend_from_slice(&is_equal_vec_trace[2 * self.air.num_filter_cols() + 1..]);
    }
}
