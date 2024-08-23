use std::{cmp::Ordering, sync::Arc};

use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::config::Com;
use p3_field::{AbstractField, PrimeField, PrimeField64};
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use super::{input_table::FilterInputTableChip, output_table::FilterOutputTableChip};
use crate::{
    common::{comp::Comp, page::Page},
    filter::page_controller::utils::compare_vecs,
};

pub mod utils;

pub struct PageController<SC: StarkGenericConfig>
where
    Val<SC>: AbstractField + PrimeField64,
{
    pub input_chip: FilterInputTableChip,
    pub output_chip: FilterOutputTableChip,

    input_chip_trace: Option<DenseMatrix<Val<SC>>>,
    input_chip_aux_trace: Option<DenseMatrix<Val<SC>>>,
    output_chip_trace: Option<DenseMatrix<Val<SC>>>,
    output_chip_aux_trace: Option<DenseMatrix<Val<SC>>>,

    input_commitment: Option<Com<SC>>,
    output_commitment: Option<Com<SC>>,

    page_traces: Vec<DenseMatrix<Val<SC>>>,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<SC: StarkGenericConfig> PageController<SC>
where
    Val<SC>: AbstractField + PrimeField64,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        range_max: u32,
        idx_limb_bits: usize,
        idx_decomp: usize,
        cmp: Comp,
    ) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus_index, range_max));
        Self {
            input_chip: FilterInputTableChip::new(
                page_bus_index,
                idx_len,
                data_len,
                start_col,
                end_col,
                idx_limb_bits,
                idx_decomp,
                range_checker.clone(),
                cmp,
            ),
            output_chip: FilterOutputTableChip::new(
                page_bus_index,
                idx_len,
                data_len,
                idx_limb_bits,
                idx_decomp,
                range_checker.clone(),
            ),
            input_chip_trace: None,
            input_chip_aux_trace: None,
            output_chip_trace: None,
            output_chip_aux_trace: None,
            input_commitment: None,
            output_commitment: None,
            page_traces: vec![],
            range_checker,
        }
    }

    pub fn input_chip_trace(&self) -> DenseMatrix<Val<SC>> {
        self.input_chip_trace.clone().unwrap()
    }

    pub fn input_chip_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.input_chip_aux_trace.clone().unwrap()
    }

    pub fn output_chip_trace(&self) -> DenseMatrix<Val<SC>> {
        self.output_chip_trace.clone().unwrap()
    }

    pub fn output_chip_aux_trace(&self) -> DenseMatrix<Val<SC>> {
        self.output_chip_aux_trace.clone().unwrap()
    }

    pub fn range_checker_trace(&self) -> DenseMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.range_checker.generate_trace()
    }

    pub fn update_range_checker(&mut self, idx_decomp: usize) {
        self.range_checker = Arc::new(RangeCheckerGateChip::new(
            self.range_checker.bus_index(),
            1 << idx_decomp,
        ));
    }

    pub fn gen_output(&self, page: Page, x: Vec<u32>, cmp: Comp) -> Page {
        let mut output: Vec<Vec<u32>> = vec![];
        let page_vec = page.to_2d_vec();

        let start_col = self.input_chip.air.start_col;
        let end_col = self.input_chip.air.end_col;

        for row in page_vec {
            // Skip the is_alloc column
            let idx_data_row = &row[1..];
            let select_cols = idx_data_row[start_col..end_col].to_vec();
            let vec_cmp = compare_vecs(select_cols, x.clone());
            match cmp {
                Comp::Lt => {
                    if vec_cmp == Ordering::Less {
                        output.push(row);
                    }
                }
                Comp::Lte => {
                    if vec_cmp != Ordering::Greater {
                        output.push(row);
                    }
                }
                Comp::Gt => {
                    if vec_cmp == Ordering::Greater {
                        output.push(row);
                    }
                }
                Comp::Gte => {
                    if vec_cmp != Ordering::Less {
                        output.push(row);
                    }
                }
                Comp::Eq => {
                    if vec_cmp == Ordering::Equal {
                        output.push(row);
                    }
                }
            }
        }

        let page_width = page.width();
        let num_remaining = page.height() - output.len();
        output.extend((0..num_remaining).map(|_| vec![0; page_width]));

        Page::from_2d_vec(&output, page.idx_len(), page.data_len())
    }
}
