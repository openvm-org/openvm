// use p3_field::PrimeField;
// use p3_matrix::dense::RowMajorMatrix;
// use p3_uni_stark::{StarkGenericConfig, Val};

// use super::air::FilterOutputTableAir;
// use crate::common::page::Page;

// impl FilterOutputTableAir {
//     /// Generates the trace of the entire page (including the is_alloc column)
//     pub fn gen_page_trace<SC: StarkGenericConfig>(&self, page: &Page) -> RowMajorMatrix<Val<SC>>
//     where
//         Val<SC>: PrimeField,
//     {
//         page.gen_trace()
//     }

//     /// Generates the auxiliary trace required to ensure proper formatting of the page
//     pub fn gen_aux_trace<SC: StarkGenericConfig>(
//         &self,
//         page: &Page,
//         range_checker: Arc<RangeCheckerGateChip>,
//     ) -> RowMajorMatrix<Val<SC>>
//     where
//         Val<SC>: PrimeField,
//     {
//         let mut rows: Vec<Vec<Val<SC>>> = vec![];

//         for i in 0..page.height() {
//             let prv_idx = if i == 0 {

//             }
//         }
//     }
// }
