use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::MyFinalPageCols, MyFinalPageAir};
use crate::sub_chip::SubAirBridge;

impl<F: PrimeField64> AirBridge<F> for MyFinalPageAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            MyFinalPageCols::<usize>::from_slice(&all_cols, self.final_air.clone());

        SubAirBridge::sends(&self.final_air, my_final_page_cols.final_page_cols)
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let my_final_page_cols =
            MyFinalPageCols::<usize>::from_slice(&all_cols, self.final_air.clone());

        // let page_cols = my_final_page_cols.final_page_cols.page_cols;

        // let page_cols = page_cols
        //     .idx
        //     .iter()
        //     .copied()
        //     .chain(page_cols.data)
        //     .map(VirtualPairCol::single_main)
        //     .collect::<Vec<_>>();

        let page_cols = [1, 2]
            .iter()
            .map(|&x| VirtualPairCol::single_main(x))
            .collect::<Vec<_>>();
        let alloc_idx = my_final_page_cols.final_page_cols.page_cols.is_alloc;
        let input_count = VirtualPairCol::single_main(alloc_idx);

        vec![Interaction {
            fields: page_cols,
            count: input_count,
            argument_index: self.page_bus_index,
        }]
    }
}
