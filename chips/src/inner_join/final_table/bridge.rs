use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::MyFinalTableAir;
use crate::final_page::columns::FinalPageCols;

impl<F: PrimeField64> AirBridge<F> for MyFinalTableAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        // Sends the same thing as FinalPageAir
        AirBridge::sends(&self.final_air)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let table_cols = FinalPageCols::<usize>::from_slice(
            &all_cols,
            self.final_air.idx_len,
            self.final_air.data_len,
            self.final_air.idx_limb_bits,
            self.final_air.idx_decomp,
        );

        let t1_cols = table_cols.page_cols.data[self.fkey_start..self.fkey_end]
            .iter()
            .chain(table_cols.page_cols.data[..self.t2_data_len].iter())
            .copied();

        let t2_cols = table_cols
            .page_cols
            .idx
            .iter()
            .chain(table_cols.page_cols.data[self.t2_data_len..].iter())
            .copied();

        vec![
            Interaction {
                fields: t1_cols.map(VirtualPairCol::single_main).collect(),
                count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                argument_index: self.t1_output_bus_index,
            },
            Interaction {
                fields: t2_cols.map(VirtualPairCol::single_main).collect(),
                count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                argument_index: self.t2_output_bus_index,
            },
        ]
    }
}
