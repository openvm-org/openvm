use crate::{
    is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIOCols},
    sub_chip::SubAirBridge,
};

use super::columns::PageIndexScanInputCols;
use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::PageIndexScanInputAir;

impl<F: PrimeField64> AirBridge<F> for PageIndexScanInputAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = PageIndexScanInputCols::<F>::get_width(
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air.limb_bits(),
            self.is_less_than_tuple_air.decomp(),
        );
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols_numbered = PageIndexScanInputCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            self.data_len,
            self.is_less_than_tuple_air.limb_bits(),
            self.is_less_than_tuple_air.decomp(),
        );

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIOCols {
                x: cols_numbered.idx.clone(),
                y: cols_numbered.x.clone(),
                tuple_less_than: cols_numbered.satisfies_pred,
            },
            aux: cols_numbered.is_less_than_tuple_aux,
        };

        // construct the row to send
        let mut cols = vec![];
        cols.push(cols_numbered.is_alloc);
        cols.extend(cols_numbered.idx);
        cols.extend(cols_numbered.data);

        let virtual_cols = cols
            .iter()
            .map(|col| VirtualPairCol::single_main(*col))
            .collect::<Vec<_>>();

        // sends with count given by send_row indicator
        let mut interactions = vec![Interaction {
            fields: virtual_cols,
            count: VirtualPairCol::single_main(cols_numbered.send_row),
            argument_index: self.bus_index,
        }];

        let mut subchip_interactions =
            SubAirBridge::<F>::sends(&self.is_less_than_tuple_air, is_less_than_tuple_cols);

        interactions.append(&mut subchip_interactions);

        interactions
    }
}
