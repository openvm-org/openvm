use super::columns::PageIndexScanVerifyCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanVerifyAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanVerifyAir {
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanVerifyCols::<F>::get_width(*self.idx_len(), *self.data_len());
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageIndexScanVerifyCols::<usize>::from_slice(
            &all_cols,
            *self.idx_len(),
            *self.data_len(),
        );

        let mut cols = vec![];
        cols.push(cols_numbered.is_alloc);
        cols.extend(cols_numbered.idx);
        cols.extend(cols_numbered.data);

        let virtual_cols = cols
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols_numbered.is_alloc),
            argument_index: *self.bus_index(),
        }]
    }
}
