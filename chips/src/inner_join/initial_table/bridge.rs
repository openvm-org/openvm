use afs_stark_backend::interaction::{AirBridge, Interaction};
use p3_air::VirtualPairCol;
use p3_field::PrimeField;

use crate::utils::to_vcols;

use super::{columns::TableCols, InitialTableAir, TableType};

impl<F: PrimeField> AirBridge<F> for InitialTableAir {
    /// For T1:
    /// - Sends idx (primary key) with multiplicity is_alloc on t1_intersector_bus (received by intersector_chip)
    /// - Sends (idx, data) with multiplicity out_mult on t1_output_bus (received by output_chip)
    /// For T2:
    /// - Sends foreign key with multiplicity is_alloc on t2_intersector_bus (received by intersector_chip)
    /// - Sends (idx, data) with multiplicity out_mult on t2_output_bus (received by output_chip)
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
                        fields: to_vcols(&table_cols.page_cols.idx),
                        count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                        argument_index: t1_intersector_bus_index,
                    },
                    Interaction {
                        fields: to_vcols(
                            &[table_cols.page_cols.idx, table_cols.page_cols.data].concat(),
                        ),
                        count: VirtualPairCol::single_main(table_cols.out_mult),
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
                        fields: to_vcols(&table_cols.page_cols.data[fkey_start..fkey_end]),
                        count: VirtualPairCol::single_main(table_cols.page_cols.is_alloc),
                        argument_index: t2_intersector_bus_index,
                    },
                    Interaction {
                        fields: to_vcols(
                            &[table_cols.page_cols.idx, table_cols.page_cols.data].concat(),
                        ),
                        count: VirtualPairCol::single_main(table_cols.out_mult),
                        argument_index: t2_output_bus_index,
                    },
                ]
            }
        }
    }

    /// For T2:
    /// - Receives foreign key with multiplicity out_mult on intersector_t2_bus (sent by intersector_chip)
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
                fields: to_vcols(&table_cols.page_cols.data[fkey_start..fkey_end]),
                count: VirtualPairCol::single_main(table_cols.out_mult),
                argument_index: intersector_t2_bus_index,
            }]
        } else {
            vec![]
        }
    }
}
