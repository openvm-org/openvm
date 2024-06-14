use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField64;

use super::{columns::TableCols, MyInitialTableAir, TableType};

impl<F: PrimeField64> AirBridge<F> for MyInitialTableAir {
    fn sends(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let table_cols = TableCols::<usize>::from_slice(&all_cols, self.idx_len, self.data_len);

        match self.table_type {
            TableType::T1 {
                t1_intersector_bus_index,
                t1_output_bus_index,
            } => {
                vec![
                    Interaction {
                        fields: table_cols
                            .idx
                            .iter()
                            .copied()
                            .map(VirtualPairCol::single_main)
                            .collect(),
                        count: VirtualPairCol::single_main(table_cols.is_alloc),
                        argument_index: t1_intersector_bus_index,
                    },
                    Interaction {
                        fields: table_cols
                            .idx
                            .iter()
                            .chain(table_cols.data.iter())
                            .copied()
                            .map(VirtualPairCol::single_main)
                            .collect(),
                        count: VirtualPairCol::single_main(table_cols.mult),
                        argument_index: t1_output_bus_index,
                    },
                ]
            }
            TableType::T2 {
                t2_intersector_bus_index,
                t2_output_bus_index,
                fkey_start,
                fkey_end,
                ..
            } => {
                vec![
                    Interaction {
                        fields: table_cols.data[fkey_start..fkey_end]
                            .iter()
                            .copied()
                            .map(VirtualPairCol::single_main)
                            .collect(),
                        count: VirtualPairCol::single_main(table_cols.is_alloc),
                        argument_index: t2_intersector_bus_index,
                    },
                    Interaction {
                        fields: table_cols
                            .idx
                            .iter()
                            .chain(table_cols.data.iter())
                            .copied()
                            .map(VirtualPairCol::single_main)
                            .collect(),
                        count: VirtualPairCol::single_main(table_cols.mult),
                        argument_index: t2_output_bus_index,
                    },
                ]
            }
        }
    }

    fn receives(&self) -> Vec<Interaction<F>> {
        let num_cols = self.air_width();
        let all_cols = (0..num_cols).collect::<Vec<usize>>();

        let table_cols = TableCols::<usize>::from_slice(&all_cols, self.idx_len, self.data_len);

        if let TableType::T2 {
            intersector_t2_bus_index,
            fkey_start,
            fkey_end,
            ..
        } = self.table_type
        {
            vec![Interaction {
                fields: table_cols.data[fkey_start..fkey_end]
                    .iter()
                    .copied()
                    .map(VirtualPairCol::single_main)
                    .collect(),
                count: VirtualPairCol::single_main(table_cols.mult),
                argument_index: intersector_t2_bus_index,
            }]
        } else {
            vec![]
        }
    }
}
