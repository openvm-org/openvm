use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use super::FinalTableAir;
use crate::{indexed_output_page_air::columns::IndexedOutputPageCols, utils::to_vcols};

impl<F: PrimeField> AirBridge<F> for FinalTableAir {
    /// Sends the same thing as FinalPageAir
    fn sends(&self) -> Vec<Interaction<F>> {
        AirBridge::sends(&self.final_air)
    }

    /// Receives (idx, data) of T1 for every allocated row on t1_output_bus (sent by t1_chip)
    /// Receives (idx, data) of T2 for every allocated row on t2_output_bus (sent by t2_chip)
    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let table_cols = IndexedOutputPageCols::<usize>::from_slice(&all_cols, &self.final_air);

        let t1_cols = table_cols.page_cols.data[self.fkey_start..self.fkey_end]
            .iter()
            .chain(table_cols.page_cols.data[self.t2_data_len..].iter())
            .copied()
            .collect::<Vec<usize>>();

        let t2_cols = table_cols
            .page_cols
            .idx
            .iter()
            .chain(table_cols.page_cols.data[..self.t2_data_len].iter())
            .copied()
            .collect::<Vec<usize>>();

        vec![
            Interaction {
                fields: to_vcols(&t1_cols),
                count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                argument_index: self.buses.t1_output_bus_index,
            },
            Interaction {
                fields: to_vcols(&t2_cols),
                count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                argument_index: self.buses.t2_output_bus_index,
            },
        ]
    }
}
