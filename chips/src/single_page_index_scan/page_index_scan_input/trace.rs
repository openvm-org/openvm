use p3_field::{AbstractField, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::sub_chip::LocalTraceInstructions;

use super::{PageIndexScanInputAir, PageIndexScanInputChip};

impl PageIndexScanInputChip {
    /// Generate the trace for the page table
    pub fn gen_page_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField,
    {
        RowMajorMatrix::new(
            page.into_iter()
                .flat_map(|row| {
                    row.into_iter()
                        .map(Val::<SC>::from_wrapped_u32)
                        .collect::<Vec<Val<SC>>>()
                })
                .collect(),
            self.page_width(),
        )
    }

    /// Generate the trace for the auxiliary columns
    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
        x: Vec<u32>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: AbstractField + PrimeField64,
    {
        let mut rows: Vec<Val<SC>> = vec![];

        for page_row in &page {
            let mut row: Vec<Val<SC>> = vec![];

            match &self.air {
                PageIndexScanInputAir::Lt {
                    idx_len,
                    is_less_than_tuple_air,
                    ..
                } => {
                    let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
                    let idx = page_row[1..1 + *idx_len].to_vec();

                    let x_trace: Vec<Val<SC>> = x
                        .iter()
                        .map(|x| Val::<SC>::from_canonical_u32(*x))
                        .collect();
                    row.extend(x_trace);

                    let is_less_than_tuple_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_less_than_tuple_air,
                            (idx.clone(), x.clone(), self.range_checker.clone()),
                        )
                        .flatten();

                    row.push(is_less_than_tuple_trace[2 * *idx_len]);
                    let send_row = is_less_than_tuple_trace[2 * *idx_len] * is_alloc;
                    row.push(send_row);

                    row.extend_from_slice(&is_less_than_tuple_trace[2 * *idx_len + 1..]);
                }
                PageIndexScanInputAir::Lte {
                    idx_len,
                    is_less_than_tuple_air,
                    is_equal_vec_air,
                    ..
                } => {
                    let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
                    let idx = page_row[1..1 + *idx_len].to_vec();

                    let x_trace: Vec<Val<SC>> = x
                        .iter()
                        .map(|x| Val::<SC>::from_canonical_u32(*x))
                        .collect();
                    row.extend(x_trace);

                    let is_less_than_tuple_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_less_than_tuple_air,
                            (idx.clone(), x.clone(), self.range_checker.clone()),
                        )
                        .flatten();

                    let is_equal_vec_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_equal_vec_air,
                            (
                                idx.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                                x.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                            ),
                        )
                        .flatten();

                    row.push(is_less_than_tuple_trace[2 * *idx_len]);
                    row.push(is_equal_vec_trace[3 * *idx_len - 1]);
                    let satisfies_pred = is_less_than_tuple_trace[2 * *idx_len]
                        + is_equal_vec_trace[3 * *idx_len - 1];
                    row.push(satisfies_pred);
                    row.push(satisfies_pred * is_alloc);

                    row.extend_from_slice(&is_less_than_tuple_trace[2 * *idx_len + 1..]);
                    row.extend_from_slice(&is_equal_vec_trace[2 * *idx_len..]);
                }
                PageIndexScanInputAir::Eq {
                    idx_len,
                    is_equal_vec_air,
                    ..
                } => {
                    let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
                    let idx = page_row[1..1 + *idx_len].to_vec();

                    let x_trace: Vec<Val<SC>> = x
                        .iter()
                        .map(|x| Val::<SC>::from_canonical_u32(*x))
                        .collect();
                    row.extend(x_trace);

                    let is_equal_vec_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_equal_vec_air,
                            (
                                idx.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                                x.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                            ),
                        )
                        .flatten();

                    row.push(is_equal_vec_trace[3 * *idx_len - 1]);
                    let send_row = is_equal_vec_trace[3 * *idx_len - 1] * is_alloc;
                    row.push(send_row);

                    row.extend_from_slice(&is_equal_vec_trace[2 * *idx_len..]);
                }
                PageIndexScanInputAir::Gte {
                    idx_len,
                    is_less_than_tuple_air,
                    is_equal_vec_air,
                    ..
                } => {
                    let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
                    let idx = page_row[1..1 + *idx_len].to_vec();

                    let x_trace: Vec<Val<SC>> = x
                        .iter()
                        .map(|x| Val::<SC>::from_canonical_u32(*x))
                        .collect();
                    row.extend(x_trace);

                    let is_less_than_tuple_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_less_than_tuple_air,
                            (x.clone(), idx.clone(), self.range_checker.clone()),
                        )
                        .flatten();

                    let is_equal_vec_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_equal_vec_air,
                            (
                                idx.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                                x.clone()
                                    .into_iter()
                                    .map(Val::<SC>::from_canonical_u32)
                                    .collect(),
                            ),
                        )
                        .flatten();

                    row.push(is_less_than_tuple_trace[2 * *idx_len]);
                    row.push(is_equal_vec_trace[3 * *idx_len - 1]);
                    let satisfies_pred = is_less_than_tuple_trace[2 * *idx_len]
                        + is_equal_vec_trace[3 * *idx_len - 1];
                    row.push(satisfies_pred);
                    row.push(satisfies_pred * is_alloc);

                    row.extend_from_slice(&is_less_than_tuple_trace[2 * *idx_len + 1..]);
                    row.extend_from_slice(&is_equal_vec_trace[2 * *idx_len..]);
                }
                PageIndexScanInputAir::Gt {
                    idx_len,
                    is_less_than_tuple_air,
                    ..
                } => {
                    let is_alloc = Val::<SC>::from_canonical_u32(page_row[0]);
                    let idx = page_row[1..1 + *idx_len].to_vec();

                    let x_trace: Vec<Val<SC>> = x
                        .iter()
                        .map(|x| Val::<SC>::from_canonical_u32(*x))
                        .collect();
                    row.extend(x_trace);

                    // we want to check if idx > x
                    let is_less_than_tuple_trace: Vec<Val<SC>> =
                        LocalTraceInstructions::generate_trace_row(
                            is_less_than_tuple_air,
                            (x.clone(), idx.clone(), self.range_checker.clone()),
                        )
                        .flatten();

                    row.push(is_less_than_tuple_trace[2 * *idx_len]);
                    let send_row = is_less_than_tuple_trace[2 * *idx_len] * is_alloc;
                    row.push(send_row);

                    row.extend_from_slice(&is_less_than_tuple_trace[2 * *idx_len + 1..]);
                }
            }

            rows.extend_from_slice(&row);
        }

        RowMajorMatrix::new(rows, self.aux_width())
    }
}
