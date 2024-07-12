use std::iter;

use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::RootSignalCols, RootSignalAir};

impl<F: PrimeField64, const COMMITMENT_LEN: usize> AirBridge<F> for RootSignalAir<COMMITMENT_LEN> {
    fn receives(&self) -> Vec<Interaction<F>> {
        vec![]
    }

    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let cols = RootSignalCols::<usize>::from_slice(
            &all_cols,
            self.idx_len,
            COMMITMENT_LEN,
            self.is_init,
        );
        if self.is_init {
            let virtual_cols = (cols.root_commitment)
                .into_iter()
                .chain(iter::once(cols.air_id))
                .map(VirtualPairCol::single_main)
                .collect::<Vec<_>>();

            vec![Interaction {
                fields: virtual_cols,
                count: VirtualPairCol::single_main(cols.mult),
                argument_index: *self.bus_index(),
            }]
        } else {
            let virtual_cols = (cols.range.clone().unwrap().0)
                .into_iter()
                .chain(cols.range.clone().unwrap().1)
                .chain(cols.root_commitment)
                .chain(iter::once(cols.air_id))
                .map(VirtualPairCol::single_main)
                .collect::<Vec<_>>();

            vec![Interaction {
                fields: virtual_cols,
                count: VirtualPairCol::single_main(cols.mult),
                argument_index: *self.bus_index(),
            }]
        }
    }
}
