use crate::sub_chip::SubAirBridge;

use super::columns::PageIndexScanOutputCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanOutputAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanOutputAir {
    // we receive the rows that satisfy the predicate
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanOutputCols::<F>::get_width(&self.final_page_air);
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered =
            PageIndexScanOutputCols::<usize>::from_slice(&all_cols, &self.final_page_air);

        let mut cols = vec![];
        cols.push(cols_numbered.final_page_cols.page_cols.is_alloc);
        cols.extend(cols_numbered.final_page_cols.page_cols.idx.clone());
        cols.extend(cols_numbered.final_page_cols.page_cols.data);

        let virtual_cols = cols
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols_numbered.final_page_cols.page_cols.is_alloc),
            argument_index: self.page_bus_index,
        }]
    }

    // we send range checks that are from the IsLessThanTuple subchip
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanOutputCols::<F>::get_width(&self.final_page_air);
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            PageIndexScanOutputCols::<usize>::from_slice(&all_cols, &self.final_page_air);

        SubAirBridge::sends(&self.final_page_air, my_final_page_cols.final_page_cols)
    }
}
