use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
};

use super::columns::PageIndexScanOutputCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanOutputAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanOutputAir {
    // we receive the rows that satisfy the predicate
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanOutputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageIndexScanOutputCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );

        let mut cols = vec![];
        cols.push(cols_numbered.is_alloc);
        cols.extend(cols_numbered.idx.clone());
        cols.extend(cols_numbered.data);

        let virtual_cols = cols
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols_numbered.is_alloc),
            argument_index: self.bus_index,
        }]
    }

    // we send range checks that are from the IsLessThanTuple subchip
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanOutputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageIndexScanOutputCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air().limb_bits().clone(),
            self.is_less_than_tuple_air().decomp(),
        );

        // range check the decompositions of x within aux columns; here the io doesn't matter
        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: cols_numbered.idx.clone(),
                y: cols_numbered.idx.clone(),
                tuple_less_than: cols_numbered.less_than_next_idx,
            },
            aux: cols_numbered.is_less_than_tuple_aux,
        };

        SubAirBridge::<F>::sends(&self.is_less_than_tuple_air, is_less_than_tuple_cols)
    }
}
